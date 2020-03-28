import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul_helper.utils import images_to_cpc_patches
from deepul_helper.resnet import ResNet18

class CPCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_dim = 64
        self.emb_scale = 0.1
        self.steps_to_ignore = 2
        self.steps_to_predict = 3

        self.encoder = ResNet18(1, use_batchnorm=False)
        self.pixelcnn = PixelCNN()

        self.z2target = nn.Conv2d(1024, self.target_dim, (1, 1))
        self.ctx2pred = nn.ModuleList([nn.Conv2d(1024, self.target_dim, (1, 1))
                                       for i in range(self.steps_to_ignore, self.steps_to_predict)])

    def forward(self, images):
        batch_size = images.shape[0]
        patches = images_to_cpc_patches(images).detach() # (N*49, C, 64, 64)
        latents = self.encoder(patches).mean(dim=[2, 3]) # (N*49, 1024)
        latents = latents.view(batch_size, 7, 7, -1).permute(0, 3, 1, 2).contiguous()
        context = self.pixelcnn(latents) # (N, 1024, 7, 7)

        col_dim, row_dim = 7, 7
        targets = self.z2target(context).view(-1, self.target_dim) # (N*49, 64)

        loss = 0.
        for i in range(self.steps_to_ignore, self.steps_to_predict):
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
        return loss

    def encode(self, images):
        batch_size = images.shape[0]
        patches = images_to_cpc_patches(images) # (N*49, C, 64, 64)
        latents = self.encoder(patches).mean(dim=[2, 3]) # (N*49, 1024)
        latents = latents.view(batch_size, 7, 7, -1) # (N, 7, 7, 1024)
        return latents.mean(dim=[1, 2])


class PixelCNN(nn.Module):
    """Following PixelCN architecture in A.2 of
       https://arxiv.org/pdf/1905.09272.pdf"""

    def __init__(self):
        super().__init__()
        latent_dim = 1024

        self.net = nn.ModuleList()
        for _ in range(5):
            block = nn.Sequential(
                nn.Conv2d(latent_dim, 256, (1, 1)),
                nn.ReLU(),
                nn.ZeroPad2d((1, 1, 0, 0)),
                nn.Conv2d(256, 256, (1, 3)),
                nn.ZeroPad2d((0, 0, 1, 0)),
                nn.Conv2d(256, 256, (2, 1)),
                nn.ReLU(),
                nn.Conv2d(256, latent_dim, (1, 1))
            )
            self.net.append(block)

    def forward(self, x):
        for block in self.net:
            x = block(x) + x
        return F.relu(x)


