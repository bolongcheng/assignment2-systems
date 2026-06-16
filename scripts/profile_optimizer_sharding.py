import pandas as pd
import torch
import torch.distributed as dist
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.ddp import DDPOverlap
from cs336_systems.optimizer_sharding import ShardedOptimizer
from scripts.benchmark_base import EVAL_ITERS, WARMUP_ITERS, get_random_data_batch, init_model
from scripts.constants import BATCH_SIZE, CONTEXT_LENGTH, MODEL_SIZES
from scripts.distributed_utils import setup


WORLD_SIZE = 2
GB = 1024**3


def ddp_forward_backward_optimize_step(
    model: DDPOverlap,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> float:
    model.reset_comm_events()
    pred = model.forward(x)
    loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    model.finish_gradient_synchronization()

    optimizer.step()

    torch.cuda.synchronize()
    comm_ms = model.communication_time_ms()
    return comm_ms


def print_memory_stats(tag):
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / GB
    reserved = torch.cuda.memory_reserved() / GB
    peak_allocated = torch.cuda.max_memory_allocated() / GB
    peak_reserved = torch.cuda.max_memory_reserved() / GB
    print(
        f"rank={dist.get_rank()}\n{tag}: \n allocated: {allocated:.2f} GB, \n reserved: {reserved:.2f} GB, \n peak_allocated: {peak_allocated:.2f} GB, \n peak_reserved: {peak_reserved:.2f} GB"
    )


def profile_ddp_with_optimizer_sharding(sharded: bool = False):
    device = torch.device(f"cuda:{dist.get_rank()}")

    torch.cuda.reset_peak_memory_stats()
    base_module = init_model(MODEL_SIZES["xl"], CONTEXT_LENGTH)
    base_module.to(device)
    ddp_model = DDPOverlap(base_module)
    print_memory_stats("after_model_init")
    torch.cuda.reset_peak_memory_stats()

    if sharded:
        optimizer = ShardedOptimizer(ddp_model.parameters(), AdamW)
    else:
        optimizer = AdamW(ddp_model.parameters())
    x, y = get_random_data_batch(CONTEXT_LENGTH, BATCH_SIZE // WORLD_SIZE)
    x, y = x.to(device), y.to(device)

    pred = ddp_model.forward(x)
    loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    ddp_model.finish_gradient_synchronization()

    print_memory_stats("before_optimizer_step")
    torch.cuda.reset_peak_memory_stats()

    optimizer.step()

    print_memory_stats("after_optimizer_step")
    torch.cuda.reset_peak_memory_stats()


def profile_worker(rank: int, world_size: int) -> None:
    setup(rank, world_size, False)

    profile_ddp_with_optimizer_sharding(sharded=True)
