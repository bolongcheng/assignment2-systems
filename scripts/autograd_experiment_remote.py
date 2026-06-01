import torch
from torch import nn

from cs336_basics.model import TransformerBlock, RotaryEmbedding

import modal


app = modal.App("cs336-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("uv")
    .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
    .uv_sync()
    .add_local_python_source("cs336_basics")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


total_size_bytes = 0


def pack_hook(t):
    if isinstance(t, nn.Parameter):
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


@app.function(
    gpu="B200",
    image=image,
    timeout=1800,
)
def experiment_remote() -> None:
    # num_layers for this model is 32
    d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
    block = TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length),
    )
    block = torch.compile(block, fullgraph=True)

    def four_blocks(x):
        x = block(x)
        x = block(x)
        x = block(x)
        x = block(x)
        return x

    x = torch.randn((4, context_length, d_model), requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = four_blocks(x)

    print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")


@app.local_entrypoint()
def main() -> None:
    experiment_remote.remote()
