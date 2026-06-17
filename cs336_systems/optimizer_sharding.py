from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.optim.optimizer import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._num_params = 0
        self._local_param_groups: list[dict[str, Any]] = []
        self._params_by_owner: dict[int, list[nn.Parameter]] = {}
        super().__init__(params, defaults={})
        # superclass constructor has already called add_param_group(params), can just use _local_param_groups directly
        self._optimizer = optimizer_cls(self._local_param_groups, **kwargs)

    def step(self, closure: Callable | None = None):
        loss = self._optimizer.step(closure)
        self._sync_parameters()
        return loss

    def _sync_parameters(self):
        for owner, params in self._params_by_owner.items():
            flat_params = _flatten_dense_tensors(params)
            dist.broadcast(flat_params, src=owner)
            unflattened_params = _unflatten_dense_tensors(flat_params, params)
            for param, unflattened_param in zip(params, unflattened_params):
                param.data.copy_(unflattened_param)

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

            owner = self._num_params % self.world_size
            self._params_by_owner.setdefault(owner, []).append(p)
            self._num_params += 1

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
