import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Some code adapted from https://github.com/sthalles/SimCLR
class SimCLR(nn.Module):
    metrics = ['Loss']
    metrics_fmt = [':.4e']

    def __init__(self, dataset, n_classes, dist):
        super().__init__()
        self.temperature = 0.5
        self.latent_dim = 2048
        self.projection_dim = 128

        # Global BatchNorm for Multi-GPU training
        resnet = models.resnet50()
        resnet = nn.SyncBatchNorm.convert_sync_batchnorm(resnet)
        self.features = []
        for name, module in resnet.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, (nn.Linear, nn.MaxPool2d)):
                self.features.append(module)
        self.features = nn.Sequential(*self.features)
        self.proj = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.projection_dim)
        )

        self.dataset = dataset
        self.n_classes = n_classes
        self.dist = dist

    def construct_classifier(self):
        return nn.Sequential(nn.BatchNorm1d(self.latent_dim, affine=False), nn.Linear(self.latent_dim, self.n_classes))

    def forward(self, images):
        n = images[0].shape[0]
        xi, xj = images
        hi, hj = self.encode(xi), self.encode(xj) # (N, latent_dim)
        zi, zj = self.proj(hi), self.proj(hj) # (N, projection_dim)
        zi, zj = F.normalize(zi), F.normalize(zj)

        # Each training example has 2N - 2 negative samples
        # 2N total samples, but exclude the current and positive sample

        zis = [torch.zeros_like(zi) for _ in range(self.dist.get_world_size())]
        zjs = [torch.zeros_like(zj) for _ in range(self.dist.get_world_size())]

        self.dist.all_gather(zis, zi)
        self.dist.all_gather(zjs, zj)

        z1 = torch.cat((zi, zj), dim=0) # (2N, projection_dim)
        z2 = torch.cat(zis + zjs, dim=0) # (2N * n_gpus, projection_dim)

        sim_matrix = torch.matmul(z1, z2.t()) # (2N, 2N * n_gpus)
        sim_matrix = sim_matrix / self.temperature
        # Mask out same-sample terms
        n_gpus = self.dist.get_world_size()
        sim_matrix[torch.arange(n), torch.arange(n)]  = -float('inf')
        sim_matrix[torch.arange(n, 2*n), torch.arange(n*n_gpus, n*n_gpus+n)] = -float('inf')

        targets = torch.cat((torch.arange(n*n_gpus, n*n_gpus+n), torch.arange(n)), dim=0)
        targets = targets.to(sim_matrix.get_device()).long()

        loss = F.cross_entropy(sim_matrix, targets, reduction='sum')
        loss = loss / (2 * n)
        return dict(Loss=loss), hi

    def encode(self, images):
        return self.features(images).squeeze()

