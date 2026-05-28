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
    model: BasicsTransformerLM,
    x: torch.Tensor,
):
    model.forward(x)
    torch.cuda.synchronize()


def forward_backward_step(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
):
    pred = model.forward(x)
    loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))
    loss.backward()
    torch.cuda.synchronize()


def forward_backward_optimize_step(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
):
    pred = model.forward(x)
    loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SIZES.keys()), type=str, default="small")
    parser.add_argument("--option", choices=list(BenchmarkOption), type=str, required=True)
    args = parser.parse_args()
    run_benchmark(args.model, args.option)


if __name__ == "__main__":
    main()
