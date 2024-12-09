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
        self.resize = configs.RESIZE if configs is not None else 0
        self.random_crop = configs.RANDOM_CROP if configs is not None else (0, 0)
        self.horizontal_flip_prob = configs.RANDOM_HORIZONTAL_FLIP_PROB if configs is not None else 0.0
        self.erasing_prob = configs.RANDOM_ERASING_PROB if configs is not None else 0.0
        self.jitter_brightness = configs.JITTER_BRIGHTNESS if configs is not None else 0.0
        self.jitter_contrast = configs.JITTER_CONTRAST if configs is not None else 0.0
        self.jitter_saturation = configs.JITTER_SATURATION if configs is not None else 0.0
        self.jitter_hue = configs.JITTER_HUE if configs is not None else 0.0
        self.color_augmentation = configs.COLOR_AUGMENTATION if configs is not None else False
        self.padding = configs.PADDING if configs is not None else 0
        self.mean = configs.NORMALIZE_MEAN if configs is not None else None  # ImageNet mean
        self.std = configs.NORMALIZE_STD if configs is not None else None  # ImageNet std

        # ================= TRAIN TRANSFORMS =================
        self.transform_train = []
        
        # Resize to specified size
        if self.resize not in [0, (0, 0)]:
            self.transform_train += [transforms.Resize(self.resize)]
            
        # Random crop to specified size
        if self.random_crop not in [0, (0, 0), None]:
            self.transform_train += [transforms.RandomCrop(self.random_crop)]
        
        # Random horizontal flip
        if (self.horizontal_flip_prob != 0.0):
            self.transform_train += [transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob)]
            
        # Color Jitter
        if (self.jitter_brightness != 0.0 or self.jitter_contrast != 0.0 or self.jitter_saturation != 0.0 or self.jitter_hue != 0.0):
            self.transform_train += [transforms.ColorJitter(brightness=self.jitter_brightness,
                                                            contrast=self.jitter_contrast,
                                                            saturation=self.jitter_saturation,
                                                            hue=self.jitter_hue)]

        # Convert to tensor
        self.transform_train += [transforms.ToTensor()]
        
        # Color augmentation
        if (self.color_augmentation):
            self.transform_train += [ColorAugmentation()]
    
        # Pad the image with the specified padding
        if self.padding not in [0, (0, 0)]:
            self.transform_train += [transforms.Pad(self.padding)]
            
        # Normalize the image with the specified mean and std
        if (self.mean is not None and self.std is not None):
            self.transform_train += [transforms.Normalize(mean=self.mean, std=self.std)]
        
        # Random erasing with specified probability
        if (self.erasing_prob != 0):
            self.transform_train += [transforms.RandomErasing(p=self.erasing_prob)]

        self.train_transform = transforms.Compose(self.transform_train)
        
        # ================= VAL TRANSFORMS =================
        self.transform_val = []

        # Resize to specified size
        if self.resize not in [0, (0, 0)]:
            self.transform_val += [transforms.Resize(self.resize)] 
            
        # Convert to tensor
        self.transform_val += [transforms.ToTensor()]

        # Normalize the image with the specified mean and std
        if (self.mean is not None and self.std is not None):
            self.transform_val += [transforms.Normalize(mean=self.mean, std=self.std)]

        self.val_transform = transforms.Compose(self.transform_val)

    def get_train_transform(self):
        return self.train_transform

    def get_val_transform(self):
        return self.val_transform