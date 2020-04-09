
class SimCLR(nn.Module):
    latent_dim = "FILL"
    metrics = ['Loss']
    metrics_fmt = [':.4e']

    def __init__(self):
        super().__init__()

    def forward(self, images):
        pass

    def encode(self, images):
        pass