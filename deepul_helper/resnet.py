import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch_norm import BatchNorm1d, BatchNorm2d

class BatchNormReLU(nn.Module):

    def __init__(self, in_features, bn_cls=BatchNorm2d, relu=True, center=True, scale=True):
        super().__init__()
        assert bn_cls in [BatchNorm1d, BatchNorm2d], 'Must use custom 1D or 2D BatchNorm'

        self.relu = relu
        self.bn = bn_cls(in_features, center=center, scale=scale)

    def forward(self, x):
        x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class Conv2dFixedPad(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=(kernel_size // 2 if stride == 1 else 0), bias=False)

        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.stride > 1:
            x = fixed_padding(x, self.kernel_size)
        return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, filters, stride, use_projection=False):
        super().__init__()
        if use_projection:
            self.proj_conv = Conv2dFixedPad(in_channels, filters, kernel_size=1, stride=stride)
            self.proj_bnr = BatchNormReLU(filters, relu=False)

        self.conv1 = Conv2dFixedPad(in_channels, filters, kernel_size=3, stride=stride)
        self.bnr1 = BatchNormReLU(filters)

        self.conv2 = Conv2dFixedPad(filters, filters, kernel_size=3, stride=1)
        self.bnr2 = BatchNormReLU(filters)

        self.use_projection = use_projection

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.proj_bnr(self.proj_conv(x))
        x = self.bnr1(self.conv1(x))
        x = self.bnr2(self.conv2(x))

        return F.relu(x + shortcut, inplace=True)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, filters, stride, use_projection=False):
        super().__init__()

        if use_projection:
            filters_out = 4 * filters
            self.proj_conv = Conv2dFixedPad(in_channels, filters_out, kernel_size=1, stride=stride)
            self.proj_bnr = BatchNormReLU(filters_out, relu=False)

        self.conv1 = Conv2dFixedPad(in_channels, filters, kernel_size=1, stride=1)
        self.bnr1 = BatchNormReLU(filters)

        self.conv2 = Conv2dFixedPad(filters, filters, kernel_size=3, stride=stride)
        self.bnr2 = BatchNormReLU(filters)

        self.conv3 = Conv2dFixedPad(filters, 4 * filters, kernel_size=1, stride=1)
        self.bnr3 = BatchNormReLU(4 * filters)

        self.use_projection = use_projection

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.proj_bnr(self.proj_conv(x))
        x = self.bnr1(self.conv1(x))
        x = self.bnr2(self.conv2(x))
        x = self.bnr3(self.conv3(x))

        return F.relu(x + shortcut, inplace=True)


class BlockGroup(nn.Module):

    def __init__(self, in_channels, filters, block_fn, blocks, stride):
        super().__init__()

        self.start_block = block_fn(in_channels, filters, stride, use_projection=True)
        if block_fn == BottleneckBlock:
            in_channels = filters * 4
        else:
            in_channels = filters

        self.blocks = []
        for _ in range(1, blocks):
            self.blocks.append(block_fn(in_channels, filters, 1))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.start_block(x)
        x = self.blocks(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block_fn, layers, width_multiplier, cifar_stem=False):
        super().__init__()

        if cifar_stem:
            self.stem = nn.Sequential(
                Conv2dFixedPad(3, 64 * width_multiplier, kernel_size=3, stride=1),
                BatchNormReLU(64 * width_multiplier)
            )
        else:
            self.stem = nn.Sequential(
                Conv2dFixedPad(3, 64 * width_multiplier, kernel_size=7, stride=2),
                BatchNormReLU(64 * width_multiplier),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.group1 = BlockGroup(64 * width_multiplier, 64 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[0], stride=1)
        self.group2 = BlockGroup(64 * width_multiplier, 128 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[1], stride=2)
        self.group3 = BlockGroup(128 * width_multiplier, 256 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[2], stride=2)
        self.group4 = BlockGroup(256 * width_multiplier, 512 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[3], stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = torch.mean(x, dim=[2, 3]).squeeze()
        return x

def resnet_v1(resnet_depth, width_multiplier, cifar_stem=False):
    model_params = {
        18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
        101: {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
        152: {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
        200: {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueERror('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    return ResNet(params['block'], params['layers'], width_multiplier, cifar_stem=cifar_stem)
