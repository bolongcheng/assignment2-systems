import modal


app = modal.App("cs336-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_dir("cs336-basics", remote_path="/root/cs336-basics")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("cs336_basics")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

bchmk_vol = modal.Volume.from_name("cs336_benchmark", create_if_missing=True)


@app.function(
    gpu="B200",
    image=image,
    volumes={"/root/benchmarks": bchmk_vol},
    timeout=1800,
)
def benchmark_remote() -> None:
    import sys

    sys.path.append("/root")
    from scripts.benchmark_base import simple_profile

    simple_profile()


@app.local_entrypoint()
def main() -> None:
    benchmark_remote.remote()
