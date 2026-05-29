import argparse
import timeit
from enum import StrEnum

import pandas as pd
import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from scripts.constants import MODEL_SIZES, BATCH_SIZE, ModelParams, VOCAB_SIZE, CONTEXT_LENGTH
from scripts.toy_model import ToyModel

AMORTIZED_NUM = 1
WARMUP_ITERS = 5
EVAL_ITERS = 10


class BenchmarkOption(StrEnum):
    FWD = "fwd"
    BWD = "bwd"
    OPT = "opt"


def init_model(model_params: ModelParams) -> BasicsTransformerLM:
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=model_params.d_model,
        d_ff=model_params.d_ff,
        num_layers=model_params.num_layers,
        num_heads=model_params.num_heads,
    )


def get_random_data_batch() -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH + 1))
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


@torch.no_grad()
def forward_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    dtype: torch.dtype = torch.float32,
):
    with torch.autocast(device_type="cuda", dtype=dtype):
        with nvtx.range("forward pass"):
            model.forward(x)
        torch.cuda.synchronize()


def forward_backward_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dtype: torch.dtype = torch.float32,
):
    with torch.autocast(device_type="cuda", dtype=dtype):
        with nvtx.range("forward pass"):
            pred = model.forward(x)

        with nvtx.range("loss"):
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))

        with nvtx.range("backward pass"):
            loss.backward()
        torch.cuda.synchronize()


def forward_backward_optimize_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    dtype: torch.dtype = torch.float32,
):
    with torch.autocast(device_type="cuda", dtype=dtype):
        with nvtx.range("forward pass"):
            pred = model.forward(x)

        with nvtx.range("loss"):
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))

        optimizer.zero_grad(set_to_none=True)

        with nvtx.range("backward pass"):
            loss.backward()

        with nvtx.range("optimizer step"):
            optimizer.step()
        torch.cuda.synchronize()


def benchmark(
    option: str,
    model_str: str,
    warmup_iters: int,
    eval_iters: int,
) -> list[float]:
    model_params = MODEL_SIZES[model_str]
    gpt = init_model(model_params)
    gpt.to(device="cuda")
    x, y = get_random_data_batch()
    x, y = x.to("cuda"), y.to("cuda")

    if option == BenchmarkOption.FWD:
        stmt = lambda: forward_step(gpt, x)
    elif option == BenchmarkOption.BWD:
        stmt = lambda: forward_backward_step(gpt, x, y)
    elif option == BenchmarkOption.OPT:
        optimizer = AdamW(gpt.parameters())
        stmt = lambda: forward_backward_optimize_step(gpt, optimizer, x, y)
    else:
        raise ValueError(f"Unknown option: {option}")

    with nvtx.range("warmup"):
        for _ in range(warmup_iters):
            stmt()

    with nvtx.range("benchmark"):
        times = timeit.repeat(stmt, repeat=eval_iters, number=AMORTIZED_NUM)
    return times


def benchmark_toy_precision(
    option: str,
    warmup_iters: int,
    eval_iters: int,
    dtype: torch.dtype = torch.float32,
) -> list[float]:
    model = ToyModel(in_features=VOCAB_SIZE, out_features=VOCAB_SIZE)
    model.to(device="cuda")
    x, y = get_random_data_batch()
    x, y = x.to("cuda"), y.to("cuda")

    if option == BenchmarkOption.FWD:
        stmt = lambda: forward_step(model, x, dtype)
    elif option == BenchmarkOption.BWD:
        stmt = lambda: forward_backward_step(model, x, y, dtype)
    elif option == BenchmarkOption.OPT:
        optimizer = AdamW(model.parameters())
        stmt = lambda: forward_backward_optimize_step(model, optimizer, x, y, dtype)
    else:
        raise ValueError(f"Unknown option: {option}")

    with torch.autocast(device_type="cuda", dtype=dtype):
        for m in model.modules():
            if hasattr(m, "weight"):
                print(f"Module: {m}, Weight dtype: {m.weight.dtype}")

    with nvtx.range("warmup"):
        for _ in range(warmup_iters):
            stmt()

    with nvtx.range("benchmark"):
        times = timeit.repeat(stmt, repeat=eval_iters, number=AMORTIZED_NUM)
    return times


def run_benchmark(model_str: str | None, option: str) -> None:
    results = {}

    if model_str is None:
        model_strs = MODEL_SIZES.keys()
        file_suffix = "all"
    else:
        model_strs = [model_str]
        file_suffix = model_str

    for model_str in model_strs:
        print(f"Benchmarking {model_str} - {option}")
        times = benchmark(option, model_str, WARMUP_ITERS, EVAL_ITERS)
        results[f"{model_str}/{option}"] = times
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(results, orient="columns")
    df.index.names = ["iteration"]
    df.to_csv(f"benchmarks/pytorch_simple_profile_{option}_warmup{WARMUP_ITERS}_{file_suffix}.csv")


def run_benchmark_toy(option: str, dtype: torch.dtype) -> None:
    results = {}

    print(f"Benchmarking toy model - {option}")
    times = benchmark_toy_precision(option, WARMUP_ITERS, EVAL_ITERS, dtype)
    results[f"toy/{option}"] = times
    torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(results, orient="columns")
    df.index.names = ["iteration"]
    df.to_csv(f"benchmarks/toy_profile_{option}_warmup{WARMUP_ITERS}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SIZES.keys()), type=str, default="small")
    parser.add_argument("--option", choices=list(BenchmarkOption), type=str, required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    run_benchmark_toy(args.option, dtype_map[args.dtype])
    # run_benchmark(args.model, args.option, dtype_map[args.dtype])


if __name__ == "__main__":
    main()
