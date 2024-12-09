import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from misc.utils import (init_pretrained_weights, weights_init_classifier,
                        weights_init_kaiming)
from reid.models.layers import GeM

class ResNet(torch.nn.Module):
    def __init__(self, model_name: str,
                 num_classes,
                 use_gem=False,
                 use_stride=False,
                 use_bottleneck=False,
                 padding_mode = 'centered',
                 pretrained=(False, None)):
        super(ResNet, self).__init__()
        
        # Variables
        self.model_name = model_name
        self.pretrained, self.url = pretrained
        self.padding_mode = padding_mode
        
        # Get the base model
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=None)
        elif self.model_name == 'resnet34':
            model = models.resnet34(weights=None)
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=None)
        elif self.model_name == 'resnet101':
            model = models.resnet101(weights=None)
        elif self.model_name == 'resnet152':
            model = models.resnet152(weights=None)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        if(self.pretrained): init_pretrained_weights(model, self.url)

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
            # Bottleneck layer based on the padding mode
            if self.padding_mode == 'centered':
                self.bottleneck = nn.BatchNorm1d(self.fc_input_shape)
            else: # 'aspect_ratio'
                self.bottleneck = nn.GroupNorm(num_groups=1, num_channels=self.fc_input_shape)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

        # The last Linear layer
        self.fc1 = torch.nn.Linear(self.fc_input_shape, num_classes, bias=False)
        self.fc1.apply(weights_init_classifier)

    # x: Should be a Tensor of size [batch_size, in_channels, height, width]
    def forward(self, x: torch.Tensor, training: bool = False):       
        embeddings = self.extractor(x) # [batch_size, 2048, 1, 1] | Up to the last Conv Layer of the last Block

        embeddings_pooled = self.pool(embeddings)                                    # [batch_size, 2048, 1, 1] | Apply Pooling 
        embeddings_pooled = embeddings_pooled.view(embeddings_pooled.shape[0], -1)   # [batch_size, 2048]

        if(self.use_bottleneck):
            features = self.bottleneck(embeddings_pooled)                           # Bottleneck layer | [batch_size, 2048]
        else:
            features = embeddings_pooled
        
        if training:
            classifications = self.fc1(features)                                    # [batch_size, num_classes]
            return embeddings_pooled, classifications
        else:
            return features

class ResNet_IBN(nn.Module):
    def __init__(self, block: nn.Module,
                 layers: list,
                 num_classes: int,
                 fc_dims: Optional[Union[List[int], Tuple[int, ...]]],
                 dropout_p: float = None,
                 use_gem: str = False,
                 use_stride: str = False,
                 use_bottleneck: str = False,
                 padding_mode = 'centered',
                 pretrained: Tuple[bool, str] = (False, None)):
        super(ResNet_IBN, self).__init__()
        
        self.scale = 64
        self.inplanes = self.scale
        self.feature_dim = 512 * block.expansion
        self.use_gem = use_gem
        self.use_stride = use_stride
        self.use_bottleneck = use_bottleneck
        self.pretrained, self.url = pretrained

        self.padding_mode = padding_mode

        self.conv1 = nn.Conv2d(3, self.scale, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, self.scale, layers[0])
        self.layer2 = self._make_layer(block, self.scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.scale * 8, layers[3], stride=1 if self.use_stride else 2)
        
        if self.use_gem:
            self.global_avgpool = GeM()
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = self._construct_fc_layer(fc_dims, self.scale * 8 * block.expansion, dropout_p)

        if self.use_bottleneck:
            # Bottleneck layer based on the padding mode
            if self.padding_mode == 'centered':
                self.bottleneck = nn.BatchNorm1d(self.feature_dim)
            else: # 'aspect_ratio'
                self.bottleneck = nn.GroupNorm(num_groups=1, num_channels=self.feature_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Load pretrained weights if specified
        if self.pretrained: init_pretrained_weights(self, self.url)       

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1):
        downsample = None
        layers = []
        ibn = False if planes == 512 else True
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
   
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims: Optional[Union[List[int], Tuple[int, ...]]], 
                            input_dim: int, 
                            dropout_p: Optional[float] = None):
        """
        Construct fully connected layer
        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))

        layers = []
        
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, training: bool = False):   
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [batch_size, 64, 85, 85]

        x = self.layer1(x)          # [batch_size, 256, 85, 85]
        x = self.layer2(x)          # [batch_size, 512, 43, 43]
        x = self.layer3(x)          # [batch_size, 1024, 22, 22]
        x = self.layer4(x)          # [batch_size, 2048, 22, 22]

        v = self.global_avgpool(x)  # [batch_size, 2048, 1, 1]
        v = v.view(v.size(0), -1)   # [batch_size, 2048]

        if self.fc is not None:
            v = self.fc(v)

        if(self.use_bottleneck):
            features = self.bottleneck(v)  # Bottleneck layer | [batch_size, 2048]
        else:
            features = v

        if training:
            classifications = self.classifier(features)  # [batch_size, num_classes]
            return v, classifications
        else:
            return features

    def load_param(self, model_path):
        try:
            param_dict = torch.load(model_path)
        except:
            param_dict = torch.load(model_path, map_location='cuda:0')
        for i in param_dict['model']:
            if 'fc' in i: continue
            self.state_dict()[i].copy_(param_dict['model'][i])