import math

import torch
import triton
import triton.language as tl


@torch.compile(dynamic=True)
def flashbackward(logsumexp_output, Q, K, V, output, grad_output, is_causal: bool = False):
    B, S_Q, d_head = Q.shape
    S = torch.einsum("bqd,bkd->bqk", Q, K) / math.sqrt(d_head)
    if is_causal:
        mask = torch.triu(torch.ones(S_Q, S_Q, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.to(torch.float32).masked_fill(mask, -1e6)
    P = torch.exp(S - logsumexp_output[:, :, None]).to(Q.dtype)
    dV = torch.einsum("bqk,bqd->bkd", P, grad_output)
    dP = torch.einsum("bqd,bkd->bqk", grad_output, V)
    D = torch.sum(output * grad_output, dim=2)  # sum over d_head
    dS = P * (dP - D[:, :, None])
    dQ = torch.einsum("bqk,bkd->bqd", dS, K) / math.sqrt(d_head)
    dK = torch.einsum("bqk,bqd->bkd", dS, Q) / math.sqrt(d_head)
    return dQ, dK, dV


class FlashAttention2Torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        q_shape = Q.shape
        assert K.shape == V.shape, "K and V must have the same shape"
        B, S_Q, d_head = q_shape
        _, S_K, _ = K.shape
        Q_rows = 16
        KV_rows = 16

        # Assuming Q, K, V sequence dimensions all divisible by 16
        Q_n_tiles = math.ceil(S_Q / Q_rows)
        KV_n_tiles = math.ceil(S_K / KV_rows)

        output = torch.empty_like(Q)
        logsumexp_output = torch.empty(q_shape[:-1], device=Q.device, dtype=torch.float32)

        for b in range(B):
            for i in range(Q_n_tiles):
                Q_tile = Q[b, i * Q_rows : (i + 1) * Q_rows, :]
                O_tile = torch.zeros((Q_rows, d_head), device=Q.device, dtype=Q.dtype)
                unnormed_softmax = torch.zeros((Q_rows,), device=Q.device, dtype=Q.dtype)
                running_max = torch.full((Q_rows,), float("-inf"), device=Q.device, dtype=torch.float32)
                for j in range(KV_n_tiles):
                    K_tile = K[b, j * KV_rows : (j + 1) * KV_rows, :]
                    V_tile = V[b, j * KV_rows : (j + 1) * KV_rows, :]
                    S = Q_tile @ K_tile.T / math.sqrt(d_head)
                    old_running_max = running_max
                    running_max = torch.maximum(old_running_max, torch.max(S, dim=1)[0])
                    P = torch.exp(S - running_max[:, None])
                    exp_mean_diff = torch.exp(old_running_max - running_max)
                    unnormed_softmax = exp_mean_diff * unnormed_softmax + torch.sum(P, dim=1)
                    O_tile = exp_mean_diff[:, None] * O_tile + P @ V_tile

                O_tile = (1 / unnormed_softmax)[:, None] * O_tile
                logsumexp_tile = running_max + torch.log(unnormed_softmax)

                output[b, i * Q_rows : (i + 1) * Q_rows, :] = O_tile
                logsumexp_output[b, i * Q_rows : (i + 1) * Q_rows] = logsumexp_tile

        ctx.save_for_backward(logsumexp_output, Q, K, V, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logsumexp_output, Q, K, V, output = ctx.saved_tensors
        dQ, dK, dV = flashbackward(logsumexp_output, Q, K, V, output, grad_output)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,  # B_q
    N_KEYS,  # B_k
    scale,  # 1/sqrt(d_head)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    Out_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    output_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    logsumexp_block = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    running_max = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    unnormed_softmax = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    high = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_positions = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
    if is_causal:
        q_end = (query_tile_index + 1) * Q_TILE_SIZE
        high = tl.cdiv(q_end, K_TILE_SIZE)

    for j in tl.range(0, high, 1):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(q_block, K_block.T) * scale
        if is_causal:
            k_positions = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            causal_mask = q_positions[:, None] >= k_positions[None, :]
            S = S + tl.where(causal_mask, 0.0, -1e6)
        old_running_max = running_max
        running_max = tl.maximum(old_running_max, tl.max(S.to(running_max.dtype), axis=1))
        P = tl.exp(S - running_max[:, None])
        unnormed_softmax = tl.exp(old_running_max - running_max) * unnormed_softmax + tl.sum(P, axis=1)
        output_block = tl.exp(old_running_max - running_max)[:, None] * output_block + tl.dot(P.to(V_block.dtype), V_block)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    output_block = (1 / unnormed_softmax)[:, None] * output_block
    logsumexp_block = running_max + tl.log(unnormed_softmax)

    tl.store(Out_block_ptr, output_block.to(Out_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, logsumexp_block.to(L_block_ptr.type.element_ty), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        B, N_QUERIES, head_dim = Q.shape
        _, N_KEYS, _ = K.shape
        assert Q.shape[0] == K.shape[0] == V.shape[0], "Batch dimensions must match"
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        logsumexp_output = torch.empty((B, N_QUERIES), dtype=torch.float32, device=Q.device)
        output = torch.empty((B, N_QUERIES, head_dim), dtype=Q.dtype, device=Q.device)

        ctx.Q_TILE_SIZE = 64 if N_QUERIES < 1024 else 128
        ctx.K_TILE_SIZE = 64
        ctx.is_causal = is_causal

        flash_fwd_kernel[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), B)](
            Q,
            K,
            V,
            output,
            logsumexp_output,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=output.stride(0),
            stride_oq=output.stride(1),
            stride_od=output.stride(2),
            stride_lb=logsumexp_output.stride(0),
            stride_lq=logsumexp_output.stride(1),
            N_QUERIES=N_QUERIES,
            N_KEYS=N_KEYS,
            scale=1 / math.sqrt(head_dim),
            D=head_dim,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, output, logsumexp_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, output, logsumexp_output = ctx.saved_tensors
        dQ, dK, dV = flashbackward(logsumexp_output, Q, K, V, output, grad_output, ctx.is_causal)
        return dQ, dK, dV, None
