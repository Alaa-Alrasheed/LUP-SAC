import torch
import torch.nn as nn
from torchvision.models import densenet121

class CIFARDenseNet121(nn.Module):
    """
    Standard torchvision DenseNet121, but structurally adapted for CIFAR-10.
    Modifies the initial layers to preserve the 32x32 spatial resolution
    which would otherwise be destroyed by ImageNet's standard 7x7 stride-2 conv.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Load standard DenseNet architecture with 10 classes
        # This automatically sizes the final linear layer (classifier) correctly
        self.densenet = densenet121(num_classes=num_classes)
        
        # 1. Replace the first 7x7 stride-2 convolution with a 3x3 stride-1 convolution
        self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. Remove the initial MaxPool2d (which aggressively downsamples) by replacing it with Identity
        self.densenet.features.pool0 = nn.Identity()

    def forward(self, x):
        return self.densenet(x)
