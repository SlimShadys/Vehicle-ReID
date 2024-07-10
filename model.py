import torch
import torchvision.models as models

class ModelBuilder:
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=1000):
        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        if self.model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif self.model_name == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2' if self.pretrained else None)
        elif self.model_name == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V2' if self.pretrained else None)
        elif self.model_name == 'resnet152':
            model = models.resnet152(weights='IMAGENET1K_V2' if self.pretrained else None)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # If num_classes is different from the default (1000), modify the final layer
        if(self.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']):
            if self.num_classes != 1000:
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, self.num_classes)

        return model
    
    def move_to(self, device):
        self.model = self.model.to(device)
        return self.model
    
    def get_number_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_number_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())