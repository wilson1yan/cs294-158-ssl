from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim


def train(model, train_loader, optimizer, epoch, device, grad_clip=None):
    model.train()

    losses = OrderedDict()
    for x in train_loader:
        if isinstance(x, list):
            x = x[0]
        x = x.to(device)
        out = model.loss(x)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v)
    return losses


def eval_loss(model, data_loader, device):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            if isinstance(x, list):
                x = x[0]
            x = x.to(device)
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)        
        
        desc = f'Test '
        for i, k in enumerate(total_losses.keys()):
            total_losses[k] /= len(data_loader.dataset)
            if i == 0:
                desc += f'{k} {total_losses[k]:.4f}'
            else:
                desc += f', {k} {total_losses[k]:.4f}'
        print(desc)


def train_epochs(model, train_loader, test_loader, device, train_args, fn=None, fn_every=1,
                 quiet=False):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = OrderedDict()
    for epoch in range(epochs):
        model.train()
        epoch_losses = train(model, train_loader, optimizer, epoch, device, grad_clip)
        for k, v in epoch_losses.items():
            if k not in losses:
                losses[k] = []
            losses[k].extend(v)

        if fn is not None and epoch % fn_every == 0:
            fn(epoch)  
    if not quiet:
        if test_loader is None:
            eval_loss(model, train_loader, device)
        else:
            eval_loss(model, test_loader, device)
    
    return losses

