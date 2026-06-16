import pandas as pd
import torch
import torch.distributed as dist
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.naive_ddp import DDP
from scripts.benchmark_base import EVAL_ITERS, WARMUP_ITERS, get_random_data_batch, init_model
from scripts.constants import BATCH_SIZE, CONTEXT_LENGTH, MODEL_SIZES
from scripts.distributed_utils import setup


WORLD_SIZE = 2


def ddp_forward_backward_optimize_step(
    model: DDP,
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
    optimizer.step()

    torch.cuda.synchronize()

    comm_ms = model.communication_time_ms()
    return comm_ms


def benchmark_ddp():
    device = torch.device(f"cuda:{dist.get_rank()}")

    base_module = init_model(MODEL_SIZES["xl"], CONTEXT_LENGTH)
    base_module.to(device)
    ddp_model = DDP(base_module)
    optimizer = AdamW(ddp_model.parameters())
    x, y = get_random_data_batch(CONTEXT_LENGTH, BATCH_SIZE // WORLD_SIZE)
    x, y = x.to(device), y.to(device)

    def stmt():
        return ddp_forward_backward_optimize_step(ddp_model, optimizer, x, y)

    for _ in range(WARMUP_ITERS):
        stmt()

    total_times_ms = []
    comm_times_ms = []

    for _ in range(EVAL_ITERS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        comm_ms = stmt()
        end_event.record()

        torch.cuda.synchronize()
        total_times_ms.append(start_event.elapsed_time(end_event))
        comm_times_ms.append(comm_ms)

    return {
        "total_ms": total_times_ms,
        "comm_ms": comm_times_ms,
    }


def worker(rank: int, world_size: int, debug: bool = True) -> None:
    setup(rank, world_size, debug)

    results = benchmark_ddp()

    if rank == 0:
        df = pd.DataFrame(
            {
                "total_ms": results["total_ms"],
                "comm_ms": results["comm_ms"],
            }
        )
        df.index.name = "iteration"
        f_name = f"benchmarks/ddp_benchmark_xl_naive_ws={world_size}_bs={BATCH_SIZE}_cl={CONTEXT_LENGTH}.csv"
        df.to_csv(f_name)
        print(f"Results saved to {f_name}")

    dist.destroy_process_group()
