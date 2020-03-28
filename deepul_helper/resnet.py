''' Taken from https://github.com/kuangliu/pytorch-cifar
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ToggleBatchNorm2d(nn.Module):
    def __init__(self, use_batchnorm, *args, **kwargs):
        super().__init__()
        if use_batchnorm:
            self._module = nn.BatchNorm2d(*args, **kwargs)
        else:
            self._module = nn.Identity()

    def forward(self, x):
        return self._module(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = ToggleBatchNorm2d(use_batchnorm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = ToggleBatchNorm2d(use_batchnorm, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                ToggleBatchNorm2d(use_batchnorm, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = ToggleBatchNorm2d(use_batchnorm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = ToggleBatchNorm2d(use_batchnorm, planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = ToggleBatchNorm2d(use_batchnorm, self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                ToggleBatchNorm2d(use_batchnorm, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channels, block, num_blocks, output_dim=None, use_batchnorm=True):
        super(ResNet, self).__init__()
        self.output_dim = output_dim
        self.use_batchnorm = use_batchnorm
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ToggleBatchNorm2d(use_batchnorm, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, num_blocks[3], stride=2)

        if output_dim is not None:
            self.linear = nn.Linear(1024*block.expansion, output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_batchnorm=self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = checkpoint(self.layer1, out)
        out = checkpoint(self.layer2, out)
        out = checkpoint(self.layer3, out)
        out = checkpoint(self.layer4, out)

        if self.output_dim is not None:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def ResNet18(input_channels, use_batchnorm=True):
    return ResNet(input_channels, BasicBlock, [2,2,2,2], use_batchnorm=use_batchnorm)

def ResNet34(input_channels, use_batchnorm=True):
    return ResNet(input_channels, BasicBlock, [3,4,6,3], use_batchnorm=use_batchnorm)

def ResNet50(input_channels, use_batchnorm=True):
    return ResNet(input_channels, Bottleneck, [3,4,6,3], use_batchnorm=use_batchnorm)

