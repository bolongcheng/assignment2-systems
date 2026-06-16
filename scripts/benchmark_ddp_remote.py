import modal


app = modal.App("cs336-benchmark-ddp")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("uv")
    .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
    .uv_sync()
    .add_local_python_source("cs336_basics", "cs336_systems")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

bchmk_vol = modal.Volume.from_name("cs336_benchmark", create_if_missing=True)


@app.function(
    gpu="A100:2",
    image=image,
    volumes={"/root/benchmarks": bchmk_vol},
    timeout=3600,
)
def benchmark_ddp_remote() -> None:
    import sys

    sys.path.append("/root")
    import torch.multiprocessing as mp

    from scripts.benchmark_ddp import WORLD_SIZE, worker

    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    bchmk_vol.commit()


@app.local_entrypoint()
def main() -> None:
    benchmark_ddp_remote.remote()
