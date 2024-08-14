import math
from torch.optim.lr_scheduler import _LRScheduler

# Custom learning rate scheduler for warmup and decay (Linear, Smooth or Cosine).
class WarmupDecayLR(_LRScheduler):
    def __init__(self, optimizer, milestones, warmup_epochs=10, gamma=0.6, cosine_power=0.45, decay_method="linear", min_lr=1.0e-6, last_epoch=-1):
        self.milestones = sorted(milestones)
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        self.decay_method = decay_method
        self.min_lr = min_lr                    # Added to control the minimum learning rate
        self.cosine_power = cosine_power        # Added to control the power of the cosine decay
        super(WarmupDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            return [max(base_lr * ((self.last_epoch + 1) / self.warmup_epochs), self.min_lr) for base_lr in self.base_lrs]

        # Decay phase
        if self.decay_method == "linear":
            return [max(self._get_linear_lr(base_lr), self.min_lr) for base_lr in self.base_lrs]
        elif self.decay_method == "smooth":
            return [max(self._get_smooth_lr(base_lr), self.min_lr) for base_lr in self.base_lrs]
        elif self.decay_method == "cosine":
            return [max(self._get_cosine_lr(base_lr), self.min_lr) for base_lr in self.base_lrs]
        else:
            raise ValueError(f"Unknown decay method: {self.decay_method}")

    # This linear decay method is the same as the one in the PyTorch implementation.
    # The only difference is that we added the `min_lr` parameter to control the minimum learning rate.
    # Plus, it also benefits from warmup.
    # If you want to not have a warmup phase and only use linear decay, you can use the PyTorch implementation (use_warmup=False in config.yml).
    def _get_linear_lr(self, base_lr):
        return max(base_lr * (self.gamma ** len([m for m in self.milestones if m <= self.last_epoch])), self.min_lr)

    def _get_smooth_lr(self, base_lr):
        if self.last_epoch < self.milestones[0]:
            return base_lr
        else:
            for i in range(len(self.milestones) - 1):
                if self.milestones[i] <= self.last_epoch < self.milestones[i+1]:
                    t = (
                        self.last_epoch - self.milestones[i]) / (self.milestones[i+1] - self.milestones[i])
                    return max(base_lr * (self.gamma ** i) * (self.gamma ** t), self.min_lr)
            return max(base_lr * (self.gamma ** len(self.milestones)), self.min_lr)

    def _get_cosine_lr(self, base_lr):
        # Assuming the last milestone is the end of the cosine decay.
        T_max = self.milestones[-1] - self.warmup_epochs

        # If the current epoch is within the warmup phase, return the base learning rate.
        if self.last_epoch < self.warmup_epochs:
            return base_lr

        # If the current epoch is within the cosine decay phase, return the cosine decay learning rate.
        elif self.last_epoch < self.milestones[-1]:
            t = (self.last_epoch - self.warmup_epochs) / T_max

            # Ensure that the learning rate does not drop below `min_lr`
            return max(base_lr * 0.5 * (1 + math.cos(math.pi * t**self.cosine_power)), self.min_lr)

        # If the current epoch is after the cosine decay phase, return the stabilized learning rate.
        else:
            return self.min_lr