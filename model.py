import torchvision.models as models
from models.resnet import ResNet

class ModelBuilder:
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=1000, model_config=None):
        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model_config = model_config
        
        if('resnet' in self.model_name):
            self.model = self.build_resnet()
        elif('vit' in self.model_name):
            raise NotImplementedError("ViT model is not implemented yet")
            self.model = self.build_vit()
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def build_resnet(self):
        # Load the proper weights
        if(self.pretrained):
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        
        # Get the base model
        if self.model_name == 'resnet18':
            base_model = models.resnet18(weights=weights)
        elif self.model_name == 'resnet34':
            base_model = models.resnet34(weights=weights)
        elif self.model_name == 'resnet50':
            base_model = models.resnet50(weights=weights)
        elif self.model_name == 'resnet101':
            base_model = models.resnet101(weights=weights)
        elif self.model_name == 'resnet152':
            base_model = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        return ResNet(base_model,
                      self.num_classes,
                      use_gem=self.model_config['use_gem'],
                      use_stride=self.model_config['use_stride'],
                      use_bottleneck=self.model_config['use_bottleneck'])
    
    def build_vit(self):
        raise NotImplementedError("ViT model is not implemented yet")
    
    def move_to(self, device):
        self.model = self.model.to(device)
        return self.model
    
    def get_number_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_number_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())