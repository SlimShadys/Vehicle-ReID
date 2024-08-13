from torch.optim.lr_scheduler import _LRScheduler

# Custom learning rate scheduler for warmup and decay (Linear or Smooth).
class WarmupDecayLR(_LRScheduler):
    def __init__(self, optimizer, milestones, warmup_epochs=10, warmup_gamma=0.6, decay_method="linear", last_epoch=-1):
        self.milestones = sorted(milestones)
        self.warmup_epochs = warmup_epochs
        self.warmup_gamma = warmup_gamma
        self.decay_method = decay_method
        super(WarmupDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
        
        # Milestone decay phase
        elif(self.decay_method == "linear"): 
            return [base_lr * (self.warmup_gamma ** len([m for m in self.milestones if m <= self.last_epoch]))
                    for base_lr in self.base_lrs]
        
        # Smooth decay phase
        else:
            return [self._get_smooth_lr(base_lr) for base_lr in self.base_lrs]

    def _get_smooth_lr(self, base_lr):
        if self.last_epoch < self.milestones[0]:
            return base_lr
        for i in range(len(self.milestones) - 1):
            if self.milestones[i] <= self.last_epoch < self.milestones[i+1]:
                t = (self.last_epoch - self.milestones[i]) / (self.milestones[i+1] - self.milestones[i])
                return base_lr * (self.warmup_gamma ** i) * (self.warmup_gamma ** t)
        return base_lr * (self.warmup_gamma ** len(self.milestones))