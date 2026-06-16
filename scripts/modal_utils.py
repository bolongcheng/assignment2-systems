import modal


BENCHMARK_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("uv")
    .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
    .uv_sync()
    .add_local_python_source("cs336_basics", "cs336_systems")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

BENCHMARK_VOLUME = modal.Volume.from_name("cs336_benchmark", create_if_missing=True)
