import torch
import torch.nn as nn
import torchvision.models as models
from utils import GeM, weights_init_kaiming, weights_init_classifier

class CustomResNet(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_classes):
        super(CustomResNet, self).__init__()

        # Extract the FC layer input shape
        self.fc_input_shape = model.fc.in_features

        # Modify the stride of the last bottleneck block'sto 1
        # Bottleneck(
        #     ...
        #     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #     (downsample): Sequential(
        #       (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #       ...
        #)
        # Modified in:
        # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
                
        # Everything except the last linear layer and AdaptiveAvgPool2d
        self.extractor = torch.nn.Sequential(*list(model.children())[:-2])

        # Generalized Mean Pooling layer
        #self.gem = GeM()
        self.gem = nn.AdaptiveAvgPool2d(1)

        # Bottleneck layer
        self.bottleneck = nn.BatchNorm1d(self.fc_input_shape) # 2048 for ResNet-50
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # The last Linear layer
        self.fc1 = torch.nn.Linear(self.fc_input_shape, num_classes, bias=False)
        self.fc1.apply(weights_init_classifier)

    # x: Should be a Tensor of size [batch_size, in_channels, height, width]
    def forward(self, x: torch.Tensor, training: bool = False):       
        embeddings = self.extractor(x) # [batch_size, 2048, 1, 1] | Up to the last Conv Layer of the last Block

        embeddings_gem = self.gem(embeddings)            # [batch_size, 2048, 1, 1] | Apply GeM pooling 
        embeddings_gem = embeddings_gem.view(embeddings_gem.shape[0], -1)    # [batch_size, 2048]

        features = self.bottleneck(embeddings_gem)       # Bottleneck layer | [batch_size, 2048]
        
        if training:
            classifications = self.fc1(features)         # [batch_size, num_classes]
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