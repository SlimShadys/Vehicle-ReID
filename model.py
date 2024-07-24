import torch
import torch.nn as nn
import torchvision.models as models
from utils import GeM

class CustomResNet(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(CustomResNet, self).__init__()

        # Extract the FC layer input shape
        self.fc_input_shape = model.fc.in_features

        # Everything except the last linear layer
        self.extractor = torch.nn.Sequential(*list(model.children())[:-1])

        # Generalized Mean Pooling layer
        self.gem = GeM()

        # Bottleneck layer
        self.bottleneck = nn.BatchNorm1d(self.fc_input_shape) # 2048 for ResNet-50
        self.bottleneck.bias.requires_grad_(False)

        # The last 2 linear layers
        self.fc1 = torch.nn.Linear(self.fc_input_shape, self.fc_input_shape, bias=False)
        self.fc2 = torch.nn.Linear(self.fc_input_shape, num_classes, bias=False)

        # Initialize the weights with a normal distribution
        nn.init.normal_(self.fc1.weight) 
        nn.init.normal_(self.fc2.weight)

        # Flatten layer
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()

    # x: Should be a Tensor of size [batch_size, in_channels, height, width]
    def forward(self, x: torch.Tensor, training: bool = False):
        embeddings = self.extractor(x) # Up to the last layer before the FC layer | [batch_size, 2048, 1, 1]

        embeddings_gem = self.gem(embeddings)            # [batch_size, 2048, 1, 1]
        embeddings_gem = self.flatten(embeddings_gem)    # [batch_size, 2048]

        features = self.bottleneck(embeddings_gem)       # Bottleneck layer | [batch_size, 2048]
        
        if training:
            classifications = self.fc1(features)         # [batch_size, 2048]
            classifications = self.relu(classifications) # [batch_size, 2048]
            classifications = self.fc2(classifications)  # [batch_size, num_classes]
            return embeddings_gem, classifications
        else:
            return features

class ModelBuilder:
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=1000):
        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        if self.model_name == 'resnet18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif self.model_name == 'resnet34':
            base_model = models.resnet34(weights='IMAGENET1K_V1' if self.pretrained else None)
        elif self.model_name == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V2' if self.pretrained else None)
        elif self.model_name == 'resnet101':
            base_model = models.resnet101(weights='IMAGENET1K_V2' if self.pretrained else None)
        elif self.model_name == 'resnet152':
            base_model = models.resnet152(weights='IMAGENET1K_V2' if self.pretrained else None)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        model = CustomResNet(base_model, self.num_classes)
        return model
    
    def move_to(self, device):
        self.model = self.model.to(device)
        return self.model
    
    def get_number_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_number_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())