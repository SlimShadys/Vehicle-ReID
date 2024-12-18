import torch.nn as nn
from reid.models.layers import Bottleneck_IBN
from reid.models.resnet import ResNet, ResNet_IBN
from reid.models.color_model import SVM, EfficientNet

resnet_urls = {
    'resnet18': "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    'resnet34': "https://download.pytorch.org/models/resnet34-b627a593.pth",
    'resnet50': "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    'resnet101': "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    'resnet152': "https://download.pytorch.org/models/resnet152-f82ba261.pth",
    'resnet50_ibn_a': "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth",
    'resnet101_ibn_a': "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth"
}

class ModelBuilder:
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=1000, model_configs=None, device='cuda'):
        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model_configs = model_configs or {}
        self.device = device

        # Define the various model builders
        # - resnet  : ResNet family
        # - vit     : Vision Transformer (ViT)
        # - efficientnetb3 : EfficientNet-B3
        # - efficientnetb5 : EfficientNet-B5
        # - svm     : Support Vector Machine (SVM)
        self.model_builders = {
            'resnet': self.build_resnet,
            'vit': self.build_vit,
            'efficientnet-b3': self.build_color_model,
            'efficientnet-b5': self.build_color_model,
            'svm': self.build_color_model,
        }

        # Supported ResNet models
        self.resnet_models = list(resnet_urls.keys())

        # Build the whole model
        self.model = self.build_model()

    # General Model Builder helper function
    def build_model(self) -> nn.Module:
        for model_type, builder in self.model_builders.items():
            if model_type in self.model_name:
                return builder()
        raise ValueError(f"Unsupported model name: {self.model_name}")

    # Specific Model Builder for ResNet family
    def build_resnet(self):
        use_gem = self.model_configs.USE_GEM # GeM or AdaptiveAvg pooling
        use_stride = self.model_configs.USE_STRIDE # Use stride in the last layer
        use_bottleneck = self.model_configs.USE_BOTTLENECK # Use Bottleneck block
        padding_mode = self.model_configs.PADDING_MODE # Padding mode
        
        if self.model_name not in self.resnet_models:
            raise ValueError(f"Unsupported ResNet model: {self.model_name}")

        if 'ibn' in self.model_name:
            if self.model_name == 'resnet50_ibn_a':
                layers = [3, 4, 6, 3]
            elif self.model_name == 'resnet101_ibn_a':
                layers = [3, 4, 23, 3]
            else:
                raise ValueError(f"Unsupported IBN model: {self.model_name}")

            return ResNet_IBN(block=Bottleneck_IBN,
                              layers=layers,
                              num_classes=self.num_classes,
                              fc_dims=None,
                              dropout_p=None,
                              use_gem=use_gem,
                              use_stride=use_stride,
                              use_bottleneck=use_bottleneck,
                              padding_mode=padding_mode,
                              pretrained=(self.pretrained, resnet_urls[self.model_name]))
        else:
            return ResNet(self.model_name,
                          self.num_classes,
                          use_gem=use_gem,
                          use_stride=use_stride,
                          use_bottleneck=use_bottleneck,
                          padding_mode=padding_mode,
                          pretrained=(self.pretrained, resnet_urls[self.model_name]))

    # Specific Model Builder for Vision Transformer
    def build_vit(self):
        raise NotImplementedError("ViT model is not implemented yet")

    def build_color_model(self):
        if self.model_name == 'efficientnet-b3' or self.model_name == 'efficientnet-b5':
            return EfficientNet(configs=self.model_configs, device=self.device)
        elif self.model_name == 'svm':
            return SVM(configs=self.model_configs)
        else:
            raise ValueError(f"Unsupported color model: {self.model_name}")
        
    def move_to(self, device):
        if isinstance(self.model, nn.Module) or isinstance(self.model, EfficientNet):
            return self.model.to(device=device)
        else:
            return self.model

    def get_number_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_number_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())