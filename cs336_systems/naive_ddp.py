import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


class DDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

        for param in self.module.parameters():
            dist.broadcast(param, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self.make_allreduce_hook())

    def make_allreduce_hook(self):
        def hook(grad):
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            return grad / dist.get_world_size()

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
