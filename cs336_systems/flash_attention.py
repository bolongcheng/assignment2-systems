import math

import torch
import triton
import triton.language as tl


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
                running_max = float("-inf")
                for j in range(KV_n_tiles):
                    K_tile = K[b, j * KV_rows : (j + 1) * KV_rows, :]
                    V_tile = V[b, j * KV_rows : (j + 1) * KV_rows, :]
                    S = Q_tile @ K_tile.T / math.sqrt(d_head)
                    old_running_max = running_max
                    running_max = torch.max(torch.tensor(old_running_max), torch.max(S, dim=1)[0])[0]
                    P = torch.exp(S - running_max)
                    unnormed_softmax = torch.exp(torch.tensor(old_running_max - running_max)) * unnormed_softmax + torch.sum(P, dim=1)
                    O_tile = torch.exp(torch.tensor(old_running_max - running_max)) * O_tile + P @ V_tile

                O_tile = (1 / unnormed_softmax)[:, None] * O_tile
                logsumexp_tile = running_max + torch.log(unnormed_softmax)

                output[b, i * Q_rows : (i + 1) * Q_rows, :] = O_tile
                logsumexp_output[b, i * Q_rows : (i + 1) * Q_rows] = logsumexp_tile

        ctx.save_for_backward(logsumexp_output, Q, K, V, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("IMPLEMENT FLASH ATTENTION FIRST!?!?")


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
        offsets=(query_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(query_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    Out_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
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

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
        output_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        running_max = float("-inf")
        running_unnormed_softmax = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            K_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            S = tl.dot(q_block, K_block.T) * scale
            old_running_max = running_max
            running_max = tl.maximum(old_running_max, tl.max(S, axis=1))
            P = tl.exp(S - running_max)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("IMPLEMENT FLASH ATTENTION FIRST!?!?")
