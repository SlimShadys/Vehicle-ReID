from typing import Optional

import torchvision.transforms as transforms

class Transformations:
    def __init__(self, dataset: Optional[str] = 'veri-776'):
        self.dataset = dataset
        
        # Specific transforms for Training
        self.train_transform = transforms.Compose([
            transforms.Resize((320, 320)),          # Resize to 320x320
            transforms.RandomCrop((320, 320)),      # Random crop to 320x320
            transforms.RandomHorizontalFlip(p=0.5), # Random horizontal flip
            transforms.Pad(10),                     # Pad the image with 10 pixels
            transforms.ToTensor(),                  # Convert to tensor
        ])

        # Specific transforms for Validation
        self.val_transform = transforms.Compose([
            transforms.Resize((320, 320)),          # Resize to 320x320
            transforms.ToTensor(),                  # Convert to tensor
        ])
        
    def get_train_transform(self):
        return self.train_transform
    
    def get_val_transform(self):
        return self.val_transform