import torch
import torch.nn as nn
import triton
from cs336_basics.model import scaled_dot_product_attention

from cs336_systems.flash_attention import FlashAttentionTriton


HEAD_DIMS = [16, 32, 64, 128]
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BATCH_SIZE = 1


class NaiveAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int, device, dtype):
        super().__init__()
        self.wo = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, q, k, v) -> torch.Tensor:
        return self.wo(scaled_dot_product_attention(q, k, v, self.mask))


configs = []
for head_dim in HEAD_DIMS:
    for mode in ["fwd", "bwd", "both"]:
        for dtype in ["fp32", "bfloat16"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["CONTEXT_LENGTH"],
                    x_vals=CONTEXT_LENGTHS,
                    line_arg="provider",
                    line_vals=["triton", "naive"],
                    line_names=["Triton", "Naive"],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="Latency (ms)",
                    plot_name=f"fused-attention-d{head_dim}-dtype{dtype}-{mode}",
                    args={
                        "HEAD_DIM": head_dim,
                        "dtype": dtype,
                        "mode": mode,
                    },
                )
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(CONTEXT_LENGTH, HEAD_DIM, dtype, mode, provider):
    assert mode in ["fwd", "bwd", "both"]
    dtype = torch.float16 if dtype == "fp32" else torch.bfloat16
    q = torch.randn((BATCH_SIZE, CONTEXT_LENGTH, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH_SIZE, CONTEXT_LENGTH, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH_SIZE, CONTEXT_LENGTH, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=True)
    do = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, HEAD_DIM, device="cuda")
    if "triton" in provider:
        atten = FlashAttentionTriton
        fn = lambda: atten.apply(q, k, v, True)
        if mode == "bwd":
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "both":
            fn = lambda: atten.apply(q, k, v, True).backward(do)
        ms = triton.testing.do_bench(fn)

    if "naive" in provider:
        atten = NaiveAttention(HEAD_DIM, CONTEXT_LENGTH, device="cuda", dtype=dtype)
        fn = lambda: atten(q, k, v)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        if mode == "both":
            fn = lambda: atten(q, k, v).backward(do)
        ms = triton.testing.do_bench(fn)

    return ms


def main(save_path: str = "benchmarks") -> None:
    bench_flash_attention.run(show_plots=True, print_data=True, save_path=save_path)


if __name__ == "__main__":
    main()
