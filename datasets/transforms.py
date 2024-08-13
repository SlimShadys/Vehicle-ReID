import random
from typing import Optional

import torch
import torchvision.transforms as transforms

class ColorAugmentation(object):
    """
    Randomly alter the intensities of RGB channels
    Reference:
    Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural Networks. NIPS 2012.
    """
    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

class Transformations:
    def __init__(self, dataset: Optional[str] = 'veri-776', configs: Optional[dict] = None):
        self.dataset = dataset
        
        # Various transformations settings
        self.height = configs['height']
        self.width = configs['width']
        self.horizontal_flip_prob = configs['random_horizontal_flip_prob']
        self.erasing_prob = configs['random_erasing_prob']
        self.jitter_brightness = configs['jitter_brightness']
        self.jitter_contrast = configs['jitter_contrast']
        self.jitter_saturation = configs['jitter_saturation']
        self.jitter_hue = configs['jitter_hue']
        self.padding = configs['padding']
        self.mean = configs['normalize_mean']   # ImageNet mean
        self.std = configs['normalize_std']     # ImageNet std
  
        # Specific transforms for Training
        self.train_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),                      # Resize to specified size
            transforms.RandomCrop((self.height, self.width)),                  # Random crop to specified size
            transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),      # Random horizontal flip
            transforms.ColorJitter(brightness=self.jitter_brightness,          # Color Jitter
                                   contrast=self.jitter_contrast,
                                   saturation=self.jitter_saturation,
                                   hue=self.jitter_hue),                     
            transforms.Pad(self.padding),                                      # Pad the image with the specified padding
            transforms.ToTensor(),                                             # Convert to tensor
            ColorAugmentation(),                                               # Color Augmentation
            transforms.Normalize(mean=self.mean,
                                 std=self.std),                                # Normalize
            transforms.RandomErasing(p=self.erasing_prob),                     # Random erasing with specified probability
        ])

        # Specific transforms for Validation
        self.val_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),                      # Resize to specified size
            transforms.ToTensor(),                                             # Convert to tensor
            transforms.Normalize(mean=self.mean,
                                 std=self.std),                                # Normalize
        ])
        
    def get_train_transform(self):
        return self.train_transform
    
    def get_val_transform(self):
        return self.val_transform