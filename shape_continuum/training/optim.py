from torch.optim.lr_scheduler import StepLR


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

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, min_lr=1e-5):
        super().__init__(optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
        self.min_lr = min_lr

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [max(group["lr"] * self.gamma, self.min_lr) for group in self.optimizer.param_groups]
