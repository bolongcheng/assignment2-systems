from dataclasses import dataclass


@dataclass
class ModelParams:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SIZES = {
    "small": ModelParams(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelParams(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelParams(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelParams(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelParams(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
