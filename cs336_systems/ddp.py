import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class CommEventsTimingMixin:
    def __init__(self):
        self._comm_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def reset_comm_events(self):
        self._comm_events.clear()

    def communication_time_ms(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self._comm_events)


class DDPNaive(nn.Module, CommEventsTimingMixin):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

        with torch.no_grad():
            for param in [*self.module.parameters(), *self.module.buffers()]:
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


class DDPFlatten(nn.Module, CommEventsTimingMixin):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

        with torch.no_grad():
            for param in [*self.module.parameters(), *self.module.buffers()]:
                dist.broadcast(param, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def allreduce_grads(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        params_with_grads = [p for p in self.module.parameters() if p.grad is not None]
        grads = [p.grad for p in params_with_grads]
        flat_grads = _flatten_dense_tensors(grads)
        start_event.record()
        dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG)
        end_event.record()
        self._comm_events.append((start_event, end_event))

        synced_grads = _unflatten_dense_tensors(flat_grads, grads)

        for param, synced_grad in zip(params_with_grads, synced_grads):
            param.grad.data.copy_(synced_grad)
