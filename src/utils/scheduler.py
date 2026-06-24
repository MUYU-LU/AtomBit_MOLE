import math
from typing import Union, Sequence, List, Optional
import mindspore as ms
from mindspore import ops
from mindspore.experimental import optim


def _to_list(x, n: int, name: str) -> List[float]:
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"{name} length must match optimizer.param_groups ({n}), but got {len(x)}.")
        return [float(v) for v in x]
    return [float(x) for _ in range(n)]


def _anneal_linear(start: float, end: float, pct: float) -> float:
    return start + (end - start) * pct


def _anneal_cos(start: float, end: float, pct: float) -> float:
    # cosine from start->end as pct:0..1
    cos_out = (1.0 + math.cos(math.pi * pct)) / 2.0
    return end + (start - end) * cos_out


def _set_lr_ref(lr_ref, value: float):
    """Set an optimizer lr object in-place when possible, preserving shared refs."""
    value = ms.Tensor(float(value), ms.float32)
    try:
        ops.assign(lr_ref, value)
        return lr_ref
    except Exception:
        if hasattr(lr_ref, "set_data"):
            lr_ref.set_data(value)
            return lr_ref
    return ms.Parameter(value)


class OneCycleLR(optim.lr_scheduler.LRScheduler):
    """
    MindSpore (experimental) OneCycleLR scheduler.

    Notes:
      - This scheduler is designed to be stepped *every iteration/batch*.
      - It sets the initial lr to max_lr / div_factor, then increases to max_lr,
        then anneals to min_lr = max_lr / (div_factor * final_div_factor).

    Args:
      optimizer: mindspore.experimental.optim.Optimizer
      max_lr: float or list[float], per param_group max lr
      total_steps: int, total number of scheduler.step() calls in the whole cycle
      pct_start: float, fraction of steps used to increase lr
      anneal_strategy: "cos" or "linear"
      div_factor: float, initial_lr = max_lr/div_factor
      final_div_factor: float, min_lr = initial_lr/final_div_factor
      three_phase: bool, whether to use 3 phases (up, down-to-initial, annihilate-to-min)
      last_epoch: int, same semantics as LRScheduler

    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: Union[float, Sequence[float]],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ):
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError(f"total_steps must be a positive int, but got {total_steps}.")
        if not (0.0 < float(pct_start) < 1.0):
            raise ValueError(f"pct_start must be in (0, 1), but got {pct_start}.")
        if div_factor <= 0 or final_div_factor <= 0:
            raise ValueError("div_factor and final_div_factor must be > 0.")
        anneal_strategy = str(anneal_strategy).lower()
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError("anneal_strategy must be 'cos' or 'linear'.")

        self.total_steps = total_steps
        self.pct_start = float(pct_start)
        self.anneal_strategy = anneal_strategy
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.three_phase = bool(three_phase)

        n_groups = len(optimizer.param_groups)
        self.max_lrs = _to_list(max_lr, n_groups, "max_lr")
        self.initial_lrs = [lr / self.div_factor for lr in self.max_lrs]
        self.min_lrs = [lr / self.final_div_factor for lr in self.initial_lrs]  # = max_lr/(div_factor*final_div_factor)

        # Keep optimizer.param_groups and optimizer.lrs pointing at the same
        # lr Parameter. AdamW reads param_groups["lr"], while MindSpore's
        # LRScheduler updates optimizer.lrs/_last_lr on newer versions.
        for i, (g, init_lr) in enumerate(zip(optimizer.param_groups, self.initial_lrs)):
            lr_ref = optimizer.lrs[i] if hasattr(optimizer, "lrs") and i < len(optimizer.lrs) else g.get("lr")
            lr_ref = _set_lr_ref(lr_ref, init_lr)
            g["lr"] = lr_ref
            if hasattr(optimizer, "lrs") and i < len(optimizer.lrs):
                optimizer.lrs[i] = lr_ref

        # phase step counts
        up_steps = int(self.total_steps * self.pct_start)
        up_steps = max(1, up_steps)

        if self.three_phase:
            # remaining split into 2 parts
            rem = self.total_steps - up_steps
            down_steps = max(1, rem // 2)
            annihilate_steps = max(1, self.total_steps - up_steps - down_steps)
            self._phase_ends = (up_steps, up_steps + down_steps, up_steps + down_steps + annihilate_steps)
        else:
            down_steps = max(1, self.total_steps - up_steps)
            self._phase_ends = (up_steps, up_steps + down_steps)
        super().__init__(optimizer, last_epoch)

    def _anneal(self, start: float, end: float, pct: float) -> float:
        pct = 0.0 if pct < 0.0 else (1.0 if pct > 1.0 else pct)
        if self.anneal_strategy == "cos":
            return _anneal_cos(start, end, pct)
        return _anneal_linear(start, end, pct)

    def get_lr(self):
        # last_epoch is "number of step() calls already executed" after base step updates it. :contentReference[oaicite:2]{index=2}
        step_num = self.last_epoch
        if step_num < 0:
            # before the very first step(), keep initial
            return list(self.initial_lrs)

        # clamp to [0, total_steps]
        if step_num >= self.total_steps:
            return list(self.min_lrs)

        if not self.three_phase:
            up_end, cycle_end = self._phase_ends
            if step_num <= up_end:
                pct = step_num / float(up_end)
                return [self._anneal(s, m, pct) for s, m in zip(self.initial_lrs, self.max_lrs)]
            # down phase: max -> min
            pct = (step_num - up_end) / float(max(1, cycle_end - up_end))
            return [self._anneal(m, mn, pct) for m, mn in zip(self.max_lrs, self.min_lrs)]

        # three_phase
        up_end, down_end, cycle_end = self._phase_ends
        if step_num <= up_end:
            pct = step_num / float(up_end)
            return [self._anneal(s, m, pct) for s, m in zip(self.initial_lrs, self.max_lrs)]
        if step_num <= down_end:
            pct = (step_num - up_end) / float(max(1, down_end - up_end))
            # max -> initial
            return [self._anneal(m, s, pct) for s, m in zip(self.initial_lrs, self.max_lrs)]
        # annihilate: initial -> min
        pct = (step_num - down_end) / float(max(1, cycle_end - down_end))
        return [self._anneal(s, mn, pct) for s, mn in zip(self.initial_lrs, self.min_lrs)]

