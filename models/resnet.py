import torch
import torch.nn as nn
from misc.utils import GeM, weights_init_classifier, weights_init_kaiming

class ResNet(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_classes, use_gem=False, use_stride=False, use_bottleneck=False):
        super(ResNet, self).__init__()
        
        # Variables
        self.num_classes = num_classes
        self.use_gem = use_gem
        self.use_stride = use_stride
        self.use_bottleneck = use_bottleneck
        
        # Extract the FC layer input shape
        self.fc_input_shape = model.fc.in_features

        if self.use_stride:
            # Modify the stride of the last bottleneck block to 1
            # Bottleneck(
            #     ...
            #     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #       ...                                                 -----
            #     (downsample): Sequential(
            #       (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            #                                                           -----
            #       ...
            #)
            # Modified in:
            # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            model.layer4[0].conv2.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)
                
        # Everything except the last linear layer and AdaptiveAvgPool2d
        self.extractor = torch.nn.Sequential(*list(model.children())[:-2])

        # Whether to use GeM pooling or Standard AdaptiveAvgPool2d
        if self.use_gem:
            self.pool = GeM() # Generalized Mean Pooling layer
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        if self.use_bottleneck:
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

        embeddings_gem = self.pool(embeddings)                              # [batch_size, 2048, 1, 1] | Apply Pooling 
        embeddings_gem = embeddings_gem.view(embeddings_gem.shape[0], -1)   # [batch_size, 2048]

        if(self.use_bottleneck):
            features = self.bottleneck(embeddings_gem)                      # Bottleneck layer | [batch_size, 2048]
        else:
            features = embeddings_gem
        
        if training:
            classifications = self.fc1(features)                            # [batch_size, num_classes]
            return embeddings_gem, classifications
        else:
            return features