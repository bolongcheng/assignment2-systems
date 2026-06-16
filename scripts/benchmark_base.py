import argparse
import timeit
from collections.abc import Callable
from contextlib import nullcontext
from enum import StrEnum

import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from scripts.constants import BATCH_SIZE, CONTEXT_LENGTH, MODEL_SIZES, VOCAB_SIZE, ModelParams
from scripts.toy_model import ToyModel


AMORTIZED_NUM = 1
WARMUP_ITERS = 5
EVAL_ITERS = 10


class BenchmarkOption(StrEnum):
    FWD = "fwd"
    BWD = "bwd"
    OPT = "opt"


def init_model(model_params: ModelParams, context_length: int) -> BasicsTransformerLM:
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=model_params.d_model,
        d_ff=model_params.d_ff,
        num_layers=model_params.num_layers,
        num_heads=model_params.num_heads,
    )


def get_random_data_batch(context_length: int = CONTEXT_LENGTH, batch_size: int = BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.randint(0, VOCAB_SIZE, (batch_size, context_length + 1))
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


# @torch.no_grad()
def forward_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    dtype: torch.dtype = torch.float32,
):
    cm = torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    with cm:
        with nvtx.range("forward pass"):
            _ = model.forward(x)
    torch.cuda.synchronize()


def forward_backward_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dtype: torch.dtype = torch.float32,
):
    cm = torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    with cm:
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
    cm = torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    with cm:
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


def create_stmt(
    model_str: str,
    option: str,
    context_length: int = CONTEXT_LENGTH,
    dtype: torch.dtype = torch.float32,
) -> Callable[[], None]:
    model_params = MODEL_SIZES[model_str]
    model = init_model(model_params, context_length=context_length)
    model.to(device="cuda")
    x, y = get_random_data_batch(context_length)
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
    return stmt


def benchmark(
    option: str,
    model_str: str,
    warmup_iters: int,
    eval_iters: int,
    dtype: torch.dtype = torch.float32,
) -> list[float]:

    stmt = create_stmt(model_str, option, CONTEXT_LENGTH, dtype)

    for _ in range(warmup_iters):
        stmt()
    times = timeit.repeat(stmt, repeat=eval_iters, number=AMORTIZED_NUM)
    return times


def profile(
    option: str,
    model_str: str,
    warmup_iters: int = WARMUP_ITERS,
    context_length: int = CONTEXT_LENGTH,
    dtype: torch.dtype = torch.float32,
) -> None:

    stmt = create_stmt(model_str, option, context_length, dtype)

    for _ in range(warmup_iters):
        stmt()
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    with nvtx.range("profile"):
        stmt()
    torch.cuda.memory._dump_snapshot(f"benchmarks/memory_snapshot_{model_str}_{option}_cl{context_length}_dtype{dtype}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)


def _print_model_dtype(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "weight"):
            print(f"Module: {m}, Weight dtype: {m.weight.dtype}")
            if m.weight.grad is not None:
                print(f"Module: {m}, Grad dtype: {m.weight.grad.dtype}")
        if hasattr(m, "bias") and m.bias is not None:
            print(f"Module: {m}, Bias dtype: {m.bias.dtype}")
            if m.bias.grad is not None:
                print(f"Module: {m}, Grad dtype: {m.bias.grad.dtype}")


def run_benchmark(
    option: str,
    model_str: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> None:
    results = {}

    if model_str is None:
        model_strs = MODEL_SIZES.keys()
        file_suffix = "all"
    else:
        model_strs = [model_str]
        file_suffix = model_str

    for model_str in model_strs:
        print(f"Benchmarking {model_str} - {option}")
        times = benchmark(option, model_str, WARMUP_ITERS, EVAL_ITERS, dtype=dtype)
        results[f"{model_str}/{option}"] = times
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(results, orient="columns")
    df.index.names = ["iteration"]
    df.to_csv(f"benchmarks/pytorch_simple_profile_{option}_warmup{WARMUP_ITERS}_{file_suffix}.csv")


def benchmark_toy_precision(
    batch_size: int = 64,
    model_out_dim: int = 36,
    dtype: torch.dtype = torch.float32,
) -> None:
    model = ToyModel(in_features=CONTEXT_LENGTH, out_features=model_out_dim)
    x = torch.rand((batch_size, CONTEXT_LENGTH))
    y = torch.randint(0, model_out_dim, (batch_size, 1))

    print("before autocast")
    _print_model_dtype(model)
    cm = torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    with cm:
        model.to(device="cuda")
        x, y = x.to("cuda"), y.to("cuda")
        pred = model(x)
        print("after forward pass")
        _print_model_dtype(model)
        loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))
    loss.backward()
    print("after backward pass")
    _print_model_dtype(model)
    print(loss.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="benchmark", choices=["benchmark", "profile", "toy"])
    parser.add_argument("--model", choices=list(MODEL_SIZES.keys()), type=str, default=None)
    parser.add_argument("--option", choices=list(BenchmarkOption), type=str, required=True)
    parser.add_argument("--dtype", default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--context_length", default=CONTEXT_LENGTH, type=int)
    args = parser.parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    if args.mode == "benchmark":
        run_benchmark(args.option, args.model, dtype=dtype_map[args.dtype])
    elif args.mode == "profile":
        profile(args.option, args.model, context_length=args.context_length)
    elif args.mode == "toy":
        benchmark_toy_precision(dtype=dtype_map[args.dtype])


if __name__ == "__main__":
    main()
