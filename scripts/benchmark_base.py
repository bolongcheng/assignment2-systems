import timeit

import numpy as np
import pandas as pd
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from scripts.constants import MODEL_SIZES, BATCH_SIZE, ModelParams, VOCAB_SIZE, CONTEXT_LENGTH

AMORTIZED_NUM = 100
WARMUP_ITERS = 5
EVAL_ITERS = 10


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
    gpt.to("cuda")
    optimizer = AdamW(gpt.parameters())
    x, y = get_random_data_batch()
    x, y = x.to("cuda"), y.to("cuda")

    if option == "forward":
        stmt = lambda: forward_step(gpt, x)
    elif option == "forward_backward":
        stmt = lambda: forward_backward_step(gpt, x, y)
    elif option == "forward_backward_optimize":
        stmt = lambda: forward_backward_optimize_step(gpt, optimizer, x, y)
    else:
        raise ValueError(f"Unknown option: {option}")

    for _ in range(warmup_iters):
        stmt()

    times = timeit.repeat(stmt, repeat=eval_iters, number=AMORTIZED_NUM)
    return times


def simple_profile() -> None:
    results = {}

    for model_str in MODEL_SIZES:
        for option in ["forward", "forward_backward", "forward_backward_optimize"]:
            print(f"Benchmarking {model_str} - {option}")
            times = benchmark(option, model_str, WARMUP_ITERS, EVAL_ITERS)
            results[f"{model_str}/{option}"] = times

    df = pd.DataFrame.from_dict(results, orient="columns")
    df.index.names = ["iteration"]
    df.to_csv("benchmarks/pytorch_simple_profile.csv")
