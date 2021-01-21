from typing import Optional

from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer


class ClippedStepLR(StepLR):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs, but not more than min_lr.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        min_lr (float): The minimum learning rate.
    """

    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1, min_lr: float = 1e-5
    ) -> None:
        super().__init__(optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
        self.min_lr = min_lr

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [max(group["lr"] * self.gamma, self.min_lr) for group in self.optimizer.param_groups]


class Trapezoid(_LRScheduler):
    """Trapezoidal scheduler.

    First warmup: Linearly increase from lower lr to max_lr, train with max_lr for most of the training.
    After 80% of iterations, start linear decline of lr.
    If annihilation, lr is linearly decreased towards an extremely small lr for the last 5% of iteratioins.
    This helps the optimizer to find the abosulte minimum at the current valey.

    Developer's note:
    if cyclic momentum would be implemented, according to Superconvergence paper
    https://arxiv.org/abs/1708.07120
    0.85 as min val works just fine. Take that value!

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        n_iterations (int): Total amount of iterations that this scheduler will be used for.
        max_lr (float): Maximimum learning rate to use.
        start_lr (float): Initial learning to use, defaults to 0.1 * max_lr.
        annihilate (bool): Whether to anneal the learning rate to 1/20 * start_lr
            after 95% of iterations have been completed.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_iterations: int,
        max_lr: float,
        start_lr: Optional[float] = None,
        annihilate: bool = True,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch)

        self.n_iters = n_iterations
        self.max_lr = max_lr
        if start_lr is None:
            self.start_lr = max_lr / 10
        else:
            self.start_lr = start_lr
        self.stop_warmup = int(0.1 * n_iterations)
        self.start_decline = int(0.8 * n_iterations)
        self.start_annihilate = int(0.95 * n_iterations) if annihilate else n_iterations

    def get_lr(self):
        if self.last_epoch < self.stop_warmup:
            step_size = (self.max_lr - self.start_lr) / self.stop_warmup
            new_lr = self.start_lr + step_size * self.last_epoch
        elif self.last_epoch < self.start_decline:
            new_lr = self.max_lr
        elif self.last_epoch <= self.start_annihilate:
            step_size = (self.max_lr - self.start_lr) / (self.start_annihilate - self.start_decline)
            new_lr = self.max_lr - step_size * (self.last_epoch - self.start_decline)
        else:
            step_size = (self.start_lr - self.start_lr / 20) / (self.n_iters - self.start_annihilate)
            new_lr = self.start_lr - step_size * (self.last_epoch - self.start_annihilate)

        return [new_lr for group in self.optimizer.param_groups]
