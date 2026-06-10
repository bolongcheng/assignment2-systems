import torch
import triton
import triton.language as tl
from einops import rearrange
from prompt_toolkit.layout import D


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row,
    x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    """
    Tiled Weighted Sum Along the Feature Dimension (Forward Pass)

    Computes:
        output[r] = sum_{d=0..D-1} x[r, d] * weight[d]
        Shape: (NUM_ROWS,)

    Parallelization Strategy:
        - Grid: 1D grid of size cdiv(NUM_ROWS, ROWS_TILE_SIZE).
        - Each program (thread block) is assigned a tile of ROWS_TILE_SIZE rows.
        - Inside each program, the feature dimension D is processed sequentially
          in blocks of D_TILE_SIZE.
        - Accumulation is done in register-resident float32 to preserve precision.

    Tiling Diagram:

                            D  (feature dim, iterated INSIDE each program)
                    ┌──────────────────────────────────────────────────────────┐
                    │  D_TILE_SIZE   D_TILE_SIZE   D_TILE_SIZE         (pad 0) │
                    │ ┌───────────┬───────────┬───────────┬─── ... ─┬────────┐ │
                    │ │   tile    │   tile    │   tile    │         │ tile 0-│ │
       ROWS_        │ │  (R x Dt) │  (R x Dt) │  (R x Dt) │         │ padded │ │
       TILE_        │ │           │           │           │         │        │ │
       SIZE   ──►   │ │   step 0  │   step 1  │   step 2  │   ...   │ step k │ │  program_id(0) = 0
            (=R)    │ └───────────┴───────────┴───────────┴─── ... ─┴────────┘ │
                    │                                                          │
                    │ ┌───────────┬───────────┬───────────┬─── ... ─┬────────┐ │
       ROWS_        │ │           │           │           │         │        │ │
       TILE_  ──►   │ │   step 0  │   step 1  │   step 2  │   ...   │ step k │ │  program_id(0) = 1
       SIZE         │ └───────────┴───────────┴───────────┴─── ... ─┴────────┘ │
                    │                                                          │
                    │                          ...                             │
                    │                                                          │
                    │ ┌───────────┬───────────┬───────────┬─── ... ─┬────────┐ │
       (last        │ │   step 0  │   step 1  │   step 2  │   ...   │ step k │ │  program_id(0) = G-1
        tile may    │ └───────────┴───────────┴───────────┴─── ... ─┴────────┘ │
        be padded   └──────────────────────────────────────────────────────────┘
        in rows)        ▲                                                  ▲
                        │       weight tile slides left → right             │
                        │ ┌───────────┬───────────┬───────────┬── ... ─┬─────┐
            weight:     │ │  Dt slice │  Dt slice │  Dt slice │        │ pad │   shape (D,), shared by all rows
                        │ └───────────┴───────────┴───────────┴── ... ─┴─────┘
                        │     step 0      step 1      step 2          step k
                        │
                        │   For each row tile r:
                        │     acc(R,) = 0
                        │     for step in 0..k:
                        │        acc += sum( x_tile(R, Dt) * weight_tile(Dt)[None,:], axis=1 )
                        │     output[r*R : (r+1)*R] = acc
                        ▼
       output (NUM_ROWS,):  one scalar per row, written once at the end.

    Legend:
      R  = ROWS_TILE_SIZE      (rows handled by ONE program / thread block)
      Dt = D_TILE_SIZE         (chunk of feature dim handled per inner-loop step)
      G  = cdiv(NUM_ROWS, R)   (grid size, = number of programs launched)
      k+1 = cdiv(D, Dt)        (number of inner-loop steps)
    """
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)
    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to load/store at a time
    # - The order of the dimensions in memory from major to minor
    # axes (= np.argsort(strides)) for optimizations, needed for
    # TMA support on >=Hopper

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide NUM_ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)
        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))  # Move by D_TILE_SIZE
    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide NUM_ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # Input
    grad_output_ptr,  # Grad input
    grad_x_ptr,
    partial_grad_weight_ptr,  # Grad outputs
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """
    Tiled Weighted Sum (Backward Pass)

    Computes:
        1) grad_x[r, d] = grad_output[r] * weight[d]
        - Pointwise / Outer-product pattern.
        - Shape: (NUM_ROWS, D)

        2) grad_weight[d] = sum_{r=0..NUM_ROWS-1} grad_output[r] * x[r, d]
        - Reduction across rows.
        - Shape: (D,)

    Reduction Strategy (Split-K / Two-Stage Reduction):
        - Row-parallelization is maintained to match the forward pass layout.
        - Since multiple programs need to accumulate into the same grad_weight[d],
        writing directly with atomics is avoided to prevent memory contention.
        - Instead, each program row-tile writes its local partial sum into a
        temporary 2D buffer `partial_grad_weight` of shape (n_row_tiles, D).
        - A quick PyTorch sum `partial_grad_weight.sum(axis=0)` is run on the GPU
        outside Triton to obtain the final grad_weight.

    Tiling Diagram:

                                    D  (inner loop walks left → right in Dt chunks)
                        ┌─────────────────────────────────────────────────────────────┐
                        │   Dt          Dt          Dt              (last tile pads) │
                        │ ┌────────┬────────┬────────┬─── ... ───┬────────┐           │
    Row-tile 0  ──►     │ │  step0 │  step1 │  step2 │           │  stepK │           │  program_id(0)=0
    (R rows of x)       │ └────────┴────────┴────────┴─── ... ───┴────────┘           │
                        │                                                              │
    Row-tile 1  ──►     │ ┌────────┬────────┬────────┬─── ... ───┬────────┐           │  program_id(0)=1
    (R rows of x)       │ └────────┴────────┴────────┴─── ... ───┴────────┘           │
                        │                       ...                                    │
    Row-tile G-1 ──►    │ ┌────────┬────────┬────────┬─── ... ───┬────────┐           │  program_id(0)=G-1
    (R rows of x)       │ └────────┴────────┴────────┴─── ... ───┴────────┘           │
                        └─────────────────────────────────────────────────────────────┘

    Inputs each step:
        x            tile  shape (R, Dt)        <- loaded
        weight       tile  shape (Dt,)          <- loaded
        grad_output  slice shape (R,)           <- loaded once (ptr never advanced)

    Outputs each step:
        grad_x       tile  shape (R, Dt)        <- stored to grad_x[row_tile, d_tile]
                            via  g[:,None] * w[None,:]
                            (each element written exactly once)

        partial_gw   tile  shape (1, Dt)        <- stored to partial_grad_weight[row_tile_idx, d_tile]
                            via  sum_r ( x_tile * g[:,None] )      <-- reduction over R only

    After kernel completes:
    partial_grad_weight  shape (G, D)
                    │
                    ▼  PyTorch:  grad_weight = partial_grad_weight.sum(axis=0)
                grad_weight  shape (D,)

    Legend:
    R  = ROWS_TILE_SIZE      (rows handled by ONE program / thread block)
    Dt = D_TILE_SIZE         (chunk of feature dim handled per inner-loop step)
    G  = cdiv(NUM_ROWS, R)   (grid size, = number of programs launched / row tiles)
    """
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(
            n_row_tiles,
            D,
        ),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE,)
        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # Never out of bounds for dim 0
        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape
        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)
        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE  # These don't have to be the same
        n_rows, D = x.shape
        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x,
            weight,
            grad_out,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight
