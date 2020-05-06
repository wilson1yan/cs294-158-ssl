import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .batch_norm import BatchNorm1d, BatchNorm2d
from .layer_norm import LayerNorm

class NormReLU(nn.Module):

    def __init__(self, input_size, relu=True, center=True, scale=True, norm_type='bn'):
        super().__init__()
        assert len(input_size) == 1 or len(input_size) == 3, f'Input size must be 1D or 3D {len(input_size)}'

        self.relu = relu
        if norm_type == 'bn':
            bn_cls = BatchNorm1d if len(input_size) == 1 else BatchNorm2d
            self.norm = bn_cls(input_size[0], center=center, scale=scale)
        elif norm_type == 'ln':
            self.norm = LayerNorm(input_size, center=center, scale=scale)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        x = self.norm(x)
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

    def __init__(self, input_size, filters, stride, use_projection=False, norm_type='bn'):
        super().__init__()

        C, H, W = input_size
        if use_projection:
            self.proj_conv = Conv2dFixedPad(C, filters, kernel_size=1, stride=stride)
            self.proj_bnr = NormReLU((filters, H // stride, W // stride),
                                      relu=False, norm_type=norm_type)

        self.conv1 = Conv2dFixedPad(C, filters, kernel_size=3, stride=stride)
        self.bnr1 = NormReLU((filters, H // stride, C // stride), norm_type=norm_type)

        self.conv2 = Conv2dFixedPad(filters, filters, kernel_size=3, stride=1)
        self.bnr2 = NormReLU((filters, H // stride, W // stride), norm_type=norm_type)

        self.use_projection = use_projection

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.proj_bnr(self.proj_conv(x))
        x = self.bnr1(self.conv1(x))
        x = self.bnr2(self.conv2(x))

        return F.relu(x + shortcut, inplace=True)


class BottleneckBlock(nn.Module):

    def __init__(self, input_size, filters, stride, use_projection=False, norm_type='bn'):
        super().__init__()

        C, H, W = input_size
        if use_projection:
            filters_out = 4 * filters
            self.proj_conv = Conv2dFixedPad(C, filters_out, kernel_size=1, stride=stride)
            self.proj_bnr = NormReLU((filters_out, H // stride, W // stride),
                                     relu=False, norm_type=norm_type)

        self.conv1 = Conv2dFixedPad(C, filters, kernel_size=1, stride=1)
        self.bnr1 = NormReLU((filters, H, W), norm_type=norm_type)

        self.conv2 = Conv2dFixedPad(filters, filters, kernel_size=3, stride=stride)
        self.bnr2 = NormReLU((filters, H // stride, W // stride), norm_type=norm_type)

        self.conv3 = Conv2dFixedPad(filters, 4 * filters, kernel_size=1, stride=1)
        self.bnr3 = NormReLU((4 * filters, H // stride, W // stride), norm_type=norm_type)

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

    def __init__(self, input_size, filters, block_fn, blocks, stride, norm_type='bn'):
        super().__init__()

        self.start_block = block_fn(input_size, filters, stride,
                                    use_projection=True, norm_type=norm_type)
        in_channels = filters * 4 if block_fn == BottleneckBlock else filters
        input_size = (4 * filters, input_size[1] // stride, input_size[2] // stride)

        self.blocks = []
        for _ in range(1, blocks):
            self.blocks.append(block_fn(input_size, filters, 1, norm_type=norm_type))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.start_block(x)
        x = self.blocks(x)
        return x


class ResNet(nn.Module):

    def __init__(self, input_size, block_fn, layers, width_multiplier, cifar_stem=False,
                 norm_type='bn'):
        super().__init__()

        C, H, W = input_size
        if cifar_stem:
            self.stem = nn.Sequential(
                Conv2dFixedPad(C, 64 * width_multiplier, kernel_size=3, stride=1),
                NormReLU((64 * width_multiplier, H, W), norm_type=norm_type)
            )
        else:
            self.stem = nn.Sequential(
                Conv2dFixedPad(C, 64 * width_multiplier, kernel_size=7, stride=2),
                NormReLU((64 * width_multiplier, H // 2, W // 2), norm_type=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            H, W = H // 4, W // 4

        scalar = 4 if block_fn == BottleneckBlock else 1

        self.group1 = BlockGroup((64 * width_multiplier, H, W), 64 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[0], stride=1,
                                 norm_type=norm_type)
        self.group2 = BlockGroup((64 * width_multiplier * scalar, H, W), 128 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[1], stride=2,
                                 norm_type=norm_type)
        H, W = H // 2, W // 2
        self.group3 = BlockGroup((128 * width_multiplier * scalar, H, W), 256 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[2], stride=2,
                                 norm_type=norm_type)
        H, W = H // 2, W // 2
        self.group4 = BlockGroup((256 * width_multiplier * scalar, H, W), 512 * width_multiplier,
                                 block_fn=block_fn, blocks=layers[3], stride=2,
                                 norm_type=norm_type)

    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = torch.mean(x, dim=[2, 3]).squeeze()
        return x

    # For semantic segmentation architectures
    def get_features(self, x):
        features = [x]

        x = self.stem[1](self.stem[0](x))
        features.append(x)

        x = self.group1(self.stem[2](x))
        features.append(x)

        x = self.group2(x)
        features.append(x)

        x = self.group3(x)
        features.append(x)

        x = self.group4(x)
        features.append(x)

        return features

def resnet_v1(input_size, resnet_depth, width_multiplier, cifar_stem=False, norm_type='bn'):
    model_params = {
        18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
        101: {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
        152: {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
        200: {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    return ResNet(input_size, params['block'], params['layers'], width_multiplier,
                  cifar_stem=cifar_stem, norm_type=norm_type)
