import modal

from scripts.modal_utils import BENCHMARK_IMAGE, BENCHMARK_VOLUME


app = modal.App("cs336-benchmark-ddp")


@app.function(
    gpu="B100:2",
    image=BENCHMARK_IMAGE,
    volumes={"/root/benchmarks": BENCHMARK_VOLUME},
    timeout=3600,
)
def benchmark_ddp_remote() -> None:
    import sys

    sys.path.append("/root")
    import torch.multiprocessing as mp

    from scripts.benchmark_ddp import WORLD_SIZE, worker

    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    BENCHMARK_VOLUME.commit()


@app.local_entrypoint()
def main() -> None:
    benchmark_ddp_remote.remote()
