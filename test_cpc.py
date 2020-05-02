import torch

from deepul_helper.tasks.cpc import PixelCNN

x = torch.randn(1, 2048, 7, 7).cuda()
x.requires_grad = True
pixelcnn = PixelCNN().cuda()

out = pixelcnn(x).sum(1).squeeze() # (7, 7)
out[3, 3].backward()

grad = x.grad
grad = torch.abs(grad).sum(1).squeeze()
print(grad)




