import modal

from scripts.modal_utils import BENCHMARK_IMAGE, BENCHMARK_VOLUME


app = modal.App("cs336-benchmark-triton")


@app.function(
    gpu="B200",
    image=BENCHMARK_IMAGE,
    volumes={"/root/benchmarks": BENCHMARK_VOLUME},
    timeout=3600,
)
def benchmark_triton_remote() -> None:
    import sys

    sys.path.append("/root")
    from scripts.benchmark_triton import main

    main(save_path="/root/benchmarks/triton")
    BENCHMARK_VOLUME.commit()


@app.local_entrypoint()
def main() -> None:
    benchmark_triton_remote.remote()
