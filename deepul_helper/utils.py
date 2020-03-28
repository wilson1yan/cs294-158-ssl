import requests
from collections import OrderedDict, Counter
import torch


def images_to_cpc_patches(images):
    """Converts (N, C, 256, 256) tensors to (N*49, C, 64, 64) patches
       for CPC training"""
    all_image_patches = []
    for r in range(7):
        for c in range(7):
            batch_patch = images[:, :, r*32:r*32+64, c*32:c*32+64]
            all_image_patches.append(batch_patch)
    # (N, 49, C, 64, 64)
    image_patches_tensor = torch.stack(all_image_patches, dim=1)
    return image_patches_tensor.view(-1, *image_patches_tensor.shape[-3:])


def quantize(img, n_bits):
    n_colors = 2 ** n_bits
    # Quantize to integers from 0, ..., n_colors - 1
    img = torch.clamp(torch.floor((img * n_colors)), max=n_colors - 1)
    img /= n_colors - 1 # Scale to [0, 1]
    return img
