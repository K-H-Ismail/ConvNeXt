import sys
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from DCLS.construct.modules.Dcls import  Dcls2d as cDcls2d
import math
import torch

_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

_cfg_cifar10 = {
    'url': '',
    'num_classes': 10, 'input_size': (3, 32, 32), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2471, 0.2435, 0.261), 'classifier': 'head'
}

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixerDcls(dim, depth, kernel_count=3, dilated_kernel_size=9, patch_size=7, n_classes=1000):
    model = nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    cDcls2d(dim, dim, kernel_count=kernel_count, dilated_kernel_size=dilated_kernel_size, 
                            groups=dim, padding=(dilated_kernel_size - 1) // 2, gain=1/(math.sqrt(dim*kernel_count))),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )
    P = torch.Tensor(2,dim, 1, kernel_count)        
    with torch.no_grad():                
        torch.nn.init.uniform_(P, -dilated_kernel_size//2, dilated_kernel_size//2)
    for i in range(depth):        
        model[i+3][0].fn[0].P = torch.nn.parameter.Parameter(P.detach().clone())
    return model

def ConvMixer(dim, depth, kernel_size=9, dilation=1, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, dilation=dilation, groups=dim, padding=dilation * (kernel_size - 1) // 2),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=9,  patch_size=1, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg    
    return model

@register_model
def convmixer_256_16_dcls(pretrained=False, **kwargs):
    model = ConvMixerDcls(256, 16, kernel_count=5,  dilated_kernel_size=9, patch_size=1, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg_cifar10    
    return model
