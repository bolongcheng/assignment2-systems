import modal

from scripts.modal_utils import BENCHMARK_IMAGE, BENCHMARK_VOLUME


NUM_GPUS = 2

app = modal.App("cs336-benchmark-distributed-communication-single-node")


@app.function(
    gpu=f"A100:{NUM_GPUS}",
    image=BENCHMARK_IMAGE,
    volumes={"/root/benchmarks": BENCHMARK_VOLUME},
    timeout=3600,
)
def benchmark_single_node_remote() -> None:
    import sys

    sys.path.append("/root")
    from scripts.distributed_communication_single_node import launch_benchmark

    launch_benchmark(world_size=NUM_GPUS, debug=False)
    BENCHMARK_VOLUME.commit()


@app.local_entrypoint()
def main() -> None:
    benchmark_single_node_remote.remote()
