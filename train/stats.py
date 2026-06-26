"""Running statistics for reward normalization (PPO / GRPO)."""

import torch
import torch.distributed as dist
from torch import nn

from .tools import TrainerTools


class RunningMeanStd(nn.Module):
    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-5):
        super().__init__()
        self.shape = shape
        self.epsilon = epsilon

        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float64))
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float64))

    def update(self, x: torch.Tensor):
        x = x.to(dtype=torch.float64)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=x.device, dtype=torch.float64)

        if TrainerTools().parallel.parallel_train:
            dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)

            batch_sum = x.sum(dim=0)
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)

            batch_sum_sq = (x ** 2).sum(dim=0)
            dist.all_reduce(batch_sum_sq, op=dist.ReduceOp.SUM)

            batch_mean = batch_sum / batch_count
            batch_mean_sq = batch_sum_sq / batch_count
            batch_var = batch_mean_sq - batch_mean ** 2

            batch_var = torch.clamp(batch_var, min=0.0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)

        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def forward(self, x: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
        target_dtype = x.dtype
        mean = self.mean.to(target_dtype)
        var = self.var.to(target_dtype)

        if shift_mean:
            return (x - mean) / torch.sqrt(var + self.epsilon)
        else:
            return x / torch.sqrt(var + self.epsilon)
