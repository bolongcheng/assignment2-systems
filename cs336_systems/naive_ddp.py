import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._comm_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self.make_allreduce_hook())

    def make_allreduce_hook(self):
        def hook(grad):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            end_event.record()
            self._comm_events.append((start_event, end_event))
            return grad / dist.get_world_size()

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def reset_comm_events(self):
        self._comm_events.clear()

    def communication_time_ms(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self._comm_events)
