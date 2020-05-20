# WIP
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul_helper.resnet import resnet_v1
from deepul_helper.batch_norm import BatchNorm1d


class CPC(nn.Module):
    latent_dim = 2048
    metrics = ['Loss']
    metrics_fmt = [':.4e']

    def __init__(self, dataset, n_classes):
        super().__init__()
        self.target_dim = 64
        self.emb_scale = 0.1
        self.steps_to_ignore = 2
        self.steps_to_predict = 3
        self.n_classes = n_classes

        self.encoder = resnet_v1((3, 64, 64), 50, 1, cifar_stem=False, norm_type='ln')
        self.pixelcnn = PixelCNN()

        self.z2target = nn.Conv2d(self.latent_dim, self.target_dim, (1, 1))
        self.ctx2pred = nn.ModuleList([nn.Conv2d(self.latent_dim, self.target_dim, (1, 1))
                                       for i in range(self.steps_to_ignore, self.steps_to_ignore + self.steps_to_predict)])

    def construct_classifier(self):
        return nn.Sequential(BatchNorm1d(self.latent_dim, center=False), nn.Linear(self.latent_dim, self.n_classes))

    def forward(self, images):
        batch_size = images.shape[0]
        patches = images_to_cpc_patches(images).detach() # (N*49, C, 64, 64)
        rnd = np.random.randint(low=0, high=16, size=(batch_size * 49,))
        for i in range(batch_size * 49):
            r, c = rnd[i] // 4, rnd[i] % 4
            patches[i, :, :r] = -1.
            patches[i, :, :, :c] = -1.
            patches[i, :, r + 60:] = -1.
            patches[i, :, :, c + 60:] = -1.

        latents = self.encoder(patches) # (N*49, latent_dim)

        latents = latents.view(batch_size, 7, 7, -1).permute(0, 3, 1, 2).contiguous() # (N, latent_dim, 7, 7)
        context = self.pixelcnn(latents) # (N, latent_dim, 7, 7)

        col_dim, row_dim = 7, 7
        targets = self.z2target(latents).permute(0, 2, 3, 1).contiguous().view(-1, self.target_dim) # (N*49, 64)

        loss = 0.
        for i in range(self.steps_to_ignore, self.steps_to_ignore + self.steps_to_predict):
            col_dim_i = col_dim - i - 1
            total_elements = batch_size * col_dim_i * row_dim

            preds_i = self.ctx2pred[i - self.steps_to_ignore](context) # (N, 64, 7, 7)
            preds_i = preds_i[:, :, :-(i+1), :] * self.emb_scale # (N, 64, H, 7)
            preds_i = preds_i.permute(0, 2, 3, 1).contiguous() # (N, H, 7, 64)
            preds_i = preds_i.view(-1, self.target_dim) # (N*H*7, 64)

            logits = torch.matmul(preds_i, targets.t()) # (N*H*7, N*49)

            b = np.arange(total_elements) // (col_dim_i * row_dim)
            col = np.arange(total_elements) % (col_dim_i * row_dim)
            labels = b * col_dim * row_dim + (i + 1) * row_dim + col
            labels = torch.LongTensor(labels).to(logits.get_device())

            loss = loss + F.cross_entropy(logits, labels)

        return dict(Loss=loss), latents.mean(dim=[2, 3])

    def encode(self, images):
        batch_size = images.shape[0]
        patches = images_to_cpc_patches(images) # (N*49, C, 64, 64)
        latents = self.encoder(patches) # (N*49, latent_dim)
        latents = latents.view(batch_size, 7, 7, -1) # (N, 7, 7, latent_dim)
        return latents.mean(dim=[1, 2])


class PixelCNN(nn.Module):
    """Following PixelCN architecture in A.2 of
       https://arxiv.org/pdf/1905.09272.pdf"""

    def __init__(self):
        super().__init__()
        latent_dim = 2048

        self.net = nn.ModuleList()
        for _ in range(5):
            block = nn.Sequential(
                nn.Conv2d(latent_dim, 256, (1, 1)),
                nn.ReLU(),
                nn.ZeroPad2d((1, 1, 0, 0)),
                nn.Conv2d(256, 256, (1, 3)),
                nn.ReLU(),
                nn.ZeroPad2d((0, 0, 1, 0)),
                nn.Conv2d(256, 256, (2, 1)),
                nn.ReLU(),
                nn.Conv2d(256, latent_dim, (1, 1))
            )
            self.net.append(block)

    def forward(self, x):
        for i, block in enumerate(self.net):
            x = block(x) + x
            x = F.relu(x)
        return x


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

# def extract_image_patches(x, kernel, stride=1, dilation=1):
#     # Do TF 'SAME' Padding
#     b,c,h,w = x.shape
#     h2 = math.ceil(h / stride)
#     w2 = math.ceil(w / stride)
#     pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
#     pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
#     x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
#     # Extract patches
#     patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
#     patches = patches.permute(0,4,5,1,2,3).contiguous()
#     return patches.view(b,-1,patches.shape[-2], patches.shape[-1])
