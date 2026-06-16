from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._local_param_groups: list[dict[str, Any]] = []
        self._param_to_rank_mapper: dict[nn.Parameter, int] = {}
        super().__init__(params, defaults={})
        # superclass constructor has already called add_param_group(params), can just use _local_param_groups directly
        self._optimizer = optimizer_cls(self._local_param_groups, **kwargs)

    def step(self, closure: Callable | None = None):
        loss = self._optimizer.step(closure)
        self._sync_parameters()
        return loss

    def _sync_parameters(self):
        for param_group in self.param_groups:
            for param in param_group["params"]:
                owner = self._param_to_rank_mapper[param]
                dist.broadcast(param.data, src=owner)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        params = param_group["params"]

        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)

        local_params = []

        for p in params:
            if not isinstance(p, nn.Parameter):
                raise TypeError(f"Expected torch.nn.Parameter, got {type(p)}")

            owner = len(self._param_to_rank_mapper) % self.world_size
            self._param_to_rank_mapper[p] = owner

            if owner == self.rank:
                local_params.append(p)

        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params

        self._local_param_groups.append(local_group)

        # If optimizer already exists (dynamic addition during training)
        if hasattr(self, "_optimizer"):
            self._optimizer.add_param_group(local_group)

        # Keep superclass bookkeeping happy
        self.param_groups.append(param_group)
