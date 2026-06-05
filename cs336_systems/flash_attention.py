import math

import torch


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
