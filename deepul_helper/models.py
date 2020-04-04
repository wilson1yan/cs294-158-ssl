import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import AlexNet

from deepul_helper.utils import images_to_cpc_patches
from deepul_helper.resnet import ResNet18

########################################## Context Encoder ####################################################

class ContextEncoder(nn.Module):
    latent_dim = 1024
    metrics = ['Loss']

    def __init__(self):
        super().__init__()
        input_channels = 3

        # Encodes the masked image
        self.encoder = nn.Sequential(
            # 128 x 128 Input
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1), # 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), # 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), # 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4) # 1 x 1
        )

        # Only reconstructs the masked part of the image and not the whole image
        self.decoder = nn.Sequential(
           nn.BatchNorm2d(1024),
           nn.ReLU(inplace=True),
           nn.ConvTranspose2d(1024, 512, 4, stride=1, padding=0), # 4 x 4
           nn.BatchNorm2d(512),
           nn.ReLU(inplace=True),
           nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 8 x 8
           nn.BatchNorm2d(256),
           nn.ReLU(inplace=True),
           nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16 x 16
           nn.BatchNorm2d(128),
           nn.ReLU(inplace=True),
           nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 32 x 32
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1), # 64 x 64
           nn.Tanh()
        )

    def forward(self, images):
        # Extract a 64 x 64 center from 128 x 128 image
        images_center = images[:, :, 32:32+64, 32:32+64].clone()
        images_masked = images.clone()
        # Mask out a 64 x 64 center with slight overlap
        images_masked[:, 0, 32+4:32+64-4, 32+4:32+64-4] = 2 * 117.0/255.0 - 1.0
        images_masked[:, 1, 32+4:32+64-4, 32+4:32+64-4] = 2 * 104.0/255.0 - 1.0
        images_masked[:, 2, 32+4:32+64-4, 32+4:32+64-4] = 2 * 123.0/255.0 - 1.0

        z = self.encoder(images_masked)
        center_recon = self.decoder(z)

        return dict(Loss=F.mse_loss(center_recon, images_center))

    def encode(self, images):
        images_masked = images
        images_masked[:, 0, 32+4:32+64-4, 32+4:32+64-4] = 2 * 117.0/255.0 - 1.0
        images_masked[:, 1, 32+4:32+64-4, 32+4:32+64-4] = 2 * 104.0/255.0 - 1.0
        images_masked[:, 2, 32+4:32+64-4, 32+4:32+64-4] = 2 * 123.0/255.0 - 1.0
        return self.encoder(images_masked).view(images.shape[0], -1)

########################################## Rotation Prediction #######################################################

class RotationPrediction(nn.Module):
    latent_dim = 256 * 6 * 6
    metrics = ['Loss', 'Acc1']

    def __init__(self):
        super().__init__()
        self.model = AlexNet(4)

    def forward(self, images):
        images, targets = self._preprocess(images)
        targets = targets.to(images.get_device())
        logits = self.model(images)
        loss = F.cross_entropy(logits, targets)

        pred = outputs.argmax(logits, dim=-1)
        correct = pred.eq(target).float().sum()
        acc = correct / targets.shape[0]

        return dict(Loss=loss, Acc1=acc)

    def encode(self, images):
        zs = self.model.features(images)
        zs = self.model.avgpool(zs)
        return zs.view(zs.shape[0], -1)

    def _preprocess(self, images):
        batch_size = images.shape[0]
        images_90 = torch.flip(images.transpose(2, 3), (2,))
        images_180 = torch.flip(images, (2, 3))
        images_270 = torch.flip(images, (2,)).transpose(2, 3)
        images_batch = torch.cat((images, images_90, images_180, images_270), dim=0)
        targets = torch.arange(4).long().repeat(batch_size)
        targets = targets.view(batch_size, 4).transpose(0, 1)
        targets = targets.contiguous().view(-1)
        return images_batch, targets

########################################## Contrastive Predictive Coding #############################################

class CPCModel(nn.Module):
    latent_dim = 1024
    metrics = ['Loss']

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
                                       for i in range(self.steps_to_ignore, self.steps_to_ignore + self.steps_to_predict)])

    def forward(self, images):
        batch_size = images.shape[0]
        patches = images_to_cpc_patches(images).detach() # (N*49, C, 64, 64)

        latents = self.encoder(patches).mean(dim=[2, 3]) # (N*49, 1024)

        latents = latents.view(batch_size, 7, 7, -1).permute(0, 3, 1, 2).contiguous()
        context = self.pixelcnn(latents) # (N, 1024, 7, 7)

        col_dim, row_dim = 7, 7
        targets = self.z2target(latents).view(-1, self.target_dim) # (N*49, 64)

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

        return dict(Loss=loss)

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
            if i < len(self.net) - 1:
                x = F.relu(x)
        return x


########################################## SimCLR ##################################################################

class SimCLR(nn.Module):
    latent_dim = "FILL"
    metrics = ['Loss']

    def __init__(self):
        super().__init__()

    def forward(self, images):
        pass

    def encode(self, images):
        pass
