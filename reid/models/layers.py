import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GeM(nn.Module):
    '''
    Generalized Mean Pooling (GeM)
    Formula: GeM = ((1/n) * sum(x^p))^(1/p)
    Taken from:
        => https://github.com/amaarora/amaarora.github.io/blob/master/nbs/GeM%20Pooling.ipynb
    '''
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = float(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)            # Input elevated to the power of p (x^p)
        avg = F.adaptive_avg_pool2d(x, 1)                # Average pooling
        return avg.pow(1. / self.p)                      # Output is elevated to the power of 1/p

class IBN(nn.Module):
    '''
    IBN (Instance-Batch Normalization) Layer
    '''
    def __init__(self, planes, use_group_norm=False):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.GroupNorm(num_groups=16, num_channels=half2) if use_group_norm else nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class Bottleneck_IBN(nn.Module):
    '''
    ResNet Bottleneck with IBN Layer instead of BatchNorm
    '''
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None, use_group_norm=False):
        super(Bottleneck_IBN, self).__init__()
        self.use_group_norm = use_group_norm
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        if ibn:
            self.bn1 = IBN(planes, self.use_group_norm)
        else:
            self.bn1 = nn.GroupNorm(num_groups=16, num_channels=planes) if self.use_group_norm else nn.BatchNorm2d(planes)
            
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=16, num_channels=planes) if self.use_group_norm else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(num_groups=16, num_channels=planes * self.expansion) if self.use_group_norm else nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
