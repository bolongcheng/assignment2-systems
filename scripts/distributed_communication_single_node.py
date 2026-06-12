import argparse
import os
import time
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


NUM_FLOAT_ELEMENTS = {
    262_144: "1 MB",
    2_621_440: "10 MB",
    26_214_400: "100 MB",
    262_144_000: "1 GB",
}

NUM_GPUS = [2, 4, 6]
WARMUP_ITERATIONS = 5
BENCHMARK_ITERATIONS = 50


def setup(rank: int, world_size: int, debug: bool = True) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if debug:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def run_distributed_timing_benchmark(
    rank: int,
    world_size: int,
    results_queue: mp.Queue,
    debug: bool = True,
) -> None:
    setup(rank, world_size, debug)

    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}")

    for num_float_elements, data_size in NUM_FLOAT_ELEMENTS.items():
        data = torch.rand(num_float_elements, dtype=torch.float32, device=device)
        print(f"Rank {rank}: All reducing {data_size} warm up")
        for _ in range(WARMUP_ITERATIONS):
            dist.all_reduce(data, async_op=False)

        if not debug:
            torch.cuda.synchronize()

        dist.barrier()

        latencies: list[float] = []
        print(f"Rank {rank}: All reducing {data_size} benchmark")
        for _ in range(BENCHMARK_ITERATIONS):
            dist.barrier()

            if not debug:
                torch.cuda.synchronize()

            start = time.perf_counter()
            dist.all_reduce(data, async_op=False)

            if not debug:
                torch.cuda.synchronize()

            latency = time.perf_counter() - start
            latencies.append(latency)

        if rank == 0:
            results_queue.put(
                {
                    data_size: latencies,
                }
            )

        dist.barrier()

    dist.destroy_process_group()


def launch_benchmark(
    world_size: int,
    debug: bool = True,
) -> None:
    with mp.Manager() as manager:
        results_queue = manager.Queue()

        mp.spawn(
            run_distributed_timing_benchmark,
            args=(world_size, results_queue, debug),
            nprocs=world_size,
            join=True,
        )

        results: list[dict[str, Any]] = []
        while not results_queue.empty():
            results.append(results_queue.get())

    combined_dict = {k: v for d in results for k, v in d.items()}
    df = pd.DataFrame(combined_dict)
    file_name = f"benchmarks/allreduce_{world_size}_node.csv"
    df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use gloo backend and CPU instead of nccl and GPU",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        choices=NUM_GPUS,
        help="Number of GPUs to use",
    )
    args = parser.parse_args()

    if args.debug:
        print("Mode          : DEBUG (gloo / CPU)")
    else:
        available_gpus = torch.cuda.device_count()
        print(f"Detected GPUs : {available_gpus}")
        if args.gpus > available_gpus:
            raise RuntimeError(f"Requested {args.gpus} GPUs but only {available_gpus} available. Use --debug to run on CPU.")

    launch_benchmark(args.gpus, args.debug)


if __name__ == "__main__":
    main()
