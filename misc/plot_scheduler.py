import math

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

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

# Set up model
lr = 1.5e-4
model = nn.Linear(100, 10)
nn.init.normal_(model.weight, mean=0, std=0.01)

# Set up epochs
epochs = 160

# Set up optimizers
linear_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
smooth_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
cosine_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
multistep_optimizer = optim.SGD(model.parameters(), lr=1.0e-3, weight_decay=0.0005, momentum=0.9, dampening=0.0, nesterov=True)

# Create scheduler
linear_scheduler = WarmupDecayLR(linear_optimizer,
                                 milestones=[20, 30, 45, 60, 75,
                                             90, 105, 120, 135, 150, 160],
                                 warmup_epochs=10,
                                 gamma=0.6,
                                 decay_method='linear',
                                 min_lr=1.0e-6,)

smooth_scheduler = WarmupDecayLR(smooth_optimizer,
                                 milestones=[20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255],
                                 warmup_epochs=10,
                                 gamma=0.6,
                                 decay_method='smooth',
                                 min_lr=1.0e-6,)

cosine_scheduler = WarmupDecayLR(cosine_optimizer,
                                 milestones=[20, 30, 45, 60, 75, 90,
                                             105, 120, 135, 150, 160, 180],
                                 warmup_epochs=10,
                                 gamma=0.6,
                                 cosine_power=0.35,
                                 decay_method='cosine',
                                 min_lr=1.0e-6,)

multistep_scheduler = WarmupDecayLR(multistep_optimizer,
                                    milestones=[20, 40, 60, 100, 120],
                                    warmup_epochs=10,
                                    gamma=0.25,
                                    decay_method='linear',
                                    min_lr=1.0e-6,)

# Collect learning rates
linear_learning_rates = []
smooth_learning_rates = []
cosine_learning_rates = []
multistep_learning_rates = []

for ep in tqdm(range(epochs), total=epochs, desc='Collecting Learning Rates'):
    linear_learning_rates.append(linear_scheduler.optimizer.param_groups[0]['lr'])
    smooth_learning_rates.append(smooth_scheduler.optimizer.param_groups[0]['lr'])
    cosine_learning_rates.append(cosine_scheduler.optimizer.param_groups[0]['lr'])
    multistep_learning_rates.append(multistep_scheduler.optimizer.param_groups[0]['lr'])

    linear_scheduler.step()
    smooth_scheduler.step()
    cosine_scheduler.step()
    multistep_scheduler.step()

# Plot the learning rates
plt.figure(figsize=(12, 6))
plt.plot(range(epochs), linear_learning_rates, label='WarmupDecayLR-Linear', color='orange')
plt.plot(range(epochs), smooth_learning_rates, label='WarmupDecayLR-Smooth', color='blue')
plt.plot(range(epochs), cosine_learning_rates, label='WarmupDecayLR-Cosine', color='purple')
plt.plot(range(epochs), multistep_learning_rates, label='MultiStepLR', color='green')
plt.title('Learning Rate Schedule')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
