from bisect import bisect_right

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

# Custom learning rate scheduler for warmup and decay.
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
       
class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, warmup_gamma=0.1, warmup_factor=0.01, warmup_iters=10, warmup_method="linear", last_epoch=-1):
        # Check if the milestones are in increasing order
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        # Check if the warmup_method is either 'constant' or 'linear'
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
            
        self.milestones = milestones
        self.warmup_gamma = warmup_gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.warmup_gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
         
# Set up your model and optimizer
model = nn.Linear(100, 10)  # Example model
smooth_optimizer = optim.Adam(model.parameters(), lr=1.5e-4)
warmup_milestone_optimizer = optim.Adam(model.parameters(), lr=1.5e-4)
warmup_multistep_optimizer = optim.Adam(model.parameters(), lr=1.5e-4)
warmup_decay_optimizer = optim.Adam(model.parameters(), lr=1.5e-4)

# Set up the scheduler
epochs = 260
steps_per_epoch = 1136  # Adjust this based on your actual steps per epoch

# Create the scheduler
smooth_scheduler = WarmupDecayLR(smooth_optimizer, 
                              milestones=[20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255],
                              warmup_epochs=10,
                              warmup_gamma=0.6,
                              decay_method='smooth')

warmup_decay = WarmupDecayLR(warmup_milestone_optimizer,
                                    milestones=[20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255],
                                    warmup_epochs=10,
                                    warmup_gamma=0.6,
                                    decay_method='linear')

warmup_multistep = WarmupMultiStepLR(warmup_multistep_optimizer,
                                    milestones=[20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255],
                                    warmup_iters=10,
                                    warmup_gamma=0.6,
                                    warmup_factor=0.03,
                                    warmup_method="linear")

# Collect learning rates
smooth_learning_rates = []
warmup_decay_learning_rates = []
warmup_multistep_learning_rates = []

for ep in tqdm(range(epochs), total=epochs, desc='Collecting Learning Rates'):
    for step in range(steps_per_epoch):
        pass
    
    smooth_learning_rates.append(smooth_scheduler.optimizer.param_groups[0]['lr'])
    warmup_decay_learning_rates.append(warmup_decay.optimizer.param_groups[0]['lr'])
    warmup_multistep_learning_rates.append(warmup_multistep.optimizer.param_groups[0]['lr'])
    
    smooth_scheduler.step()
    warmup_decay.step()
    warmup_multistep.step()
    
# Plot the learning rates
plt.figure(figsize=(12, 6))
plt.plot(range(epochs), smooth_learning_rates, label='WarmupDecayLR-Smooth', color='blue')
plt.plot(range(epochs), warmup_decay_learning_rates, label='WarmupDecayLR-Linear', color='red')
plt.plot(range(epochs), warmup_multistep_learning_rates, label='WarmupMultiStepLR', color='green')
plt.title('Learning Rate Schedule')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()