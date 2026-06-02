from itertools import product
import timeit

import modal
import pandas as pd
import torch
import torch.nn as nn

from cs336_basics.model import scaled_dot_product_attention


app = modal.App("cs336-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("uv")
    .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
    .uv_sync()
    .add_local_python_source("cs336_basics")
    .add_local_dir("scripts", remote_path="/root/scripts")
)
bchmk_vol = modal.Volume.from_name("cs336_benchmark", create_if_missing=True)

BATCH_SIZE = 8
WARMUP_ITERS = 5
EVAL_ITERS = 100
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        return self.wo(scaled_dot_product_attention(q, k, v, self.mask))


def benchmark_sdp_attention(d_model: int, seq_len: int, warmup_iter: int, eval_iters: int):
    atten = SingleHeadAttention(
        d_model=d_model,
        seq_len=seq_len,
    ).to("cuda")
    x = torch.randn((BATCH_SIZE, seq_len, d_model), device="cuda")

    def stmt():
        atten(x)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup_iter):
        stmt()

    times = timeit.repeat(stmt, repeat=eval_iters, number=1)

    # memory usage
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    print(f"Peak memory for {d_model=}, {seq_len=}: {peak_memory:.2f} MB")

    return times


@app.function(
    gpu="B200",
    image=image,
    volumes={"/root/benchmarks": bchmk_vol},
    timeout=1800,
)
def run_benchmark():
    results = {}
    for d_model, seq_len in product(D_MODELS, SEQ_LENS):
        result = benchmark_sdp_attention(d_model, seq_len, WARMUP_ITERS, EVAL_ITERS)
        results[f"{d_model}_{seq_len}"] = result

    df = pd.DataFrame.from_dict(results, orient="columns")
    df.index.names = ["iteration"]
    df.to_csv("/root/benchmarks/sdp_attention_time.csv")


@app.local_entrypoint()
def main():
    run_benchmark.remote()
