import os.path as osp

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision.utils import make_grid

from deepul_helper.data import get_datasets
from deepul_helper.tasks import *
from deepul_helper.utils import accuracy, unnormalize


def load_model_and_data(task):
    train_dset, test_dset, n_classes = get_datasets('cifar10', task)
    train_loader = data.DataLoader(train_dset, batch_size=128, num_workers=4,
                                   pin_memory=True)
    test_loader = data.DataLoader(test_dset, batch_size=128, num_workers=4,
                                  pin_memory=True)

    ckpt_pth = osp.join('results', f'cifar10_{task}', 'model_best.pth.tar')
    ckpt = torch.load(ckpt_pth, map_location='cpu')

    if task == 'context_encoder':
        model = ContextEncoder('cifar10', n_classes)
    elif task == 'rotation':
        model = RotationPrediction('cifar10', n_classes)
    elif task == 'simclr':
        model = SimCLR('cifar10', n_classes, None)
    model.load_state_dict(ckpt['state_dict'])

    model.cuda()
    model.eval()

    linear_classifier = model.construct_classifier()
    linear_classifier.load_state_dict(ckpt['state_dict_linear'])

    linear_classifier.cuda()
    linear_classifier.eval()

    return model, linear_classifier, train_loader, test_loader


def evaluate_accuracy(model, linear_classifier, train_loader, test_loader):
    train_acc1, train_acc5 = evaluate_classifier(model, linear_classifier, train_loader)
    test_acc1, test_acc5 = evaluate_classifier(model, linear_classifier, test_loader)

    print('Train Set')
    print(f'Top 1 Accuracy: {train_acc1}, Top 5 Accuracy: {train_acc5}\n')
    print('Test Set')
    print(f'Top 1 Accuracy: {test_acc1}, Top 5 Accuracy: {test_acc5}\n')


def evaluate_classifier(model, linear_classifier, loader):
    correct1, correct5 = 0, 0
    with torch.no_grad():
        for images, target in loader:
            images = images_to_cuda(images)
            target = target.cuda(non_blocking=True)
            out, zs = model(images)

            logits = linear_classifier(zs)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            correct1 += acc1.item() * bs
            correct5 += acc5.item() * bs
    total = len(loader.dataset)

    return correct1 / total, correct5 / total


def display_nearest_neighbors(model, loader, n_examples=4, k=10):
    with torch.no_grad():
        all_images, all_zs = [], []
        for i, (images, _) in loader:
            images = images_to_cuda(images)
            if i == 0:
                zs = model.encode(images)
                ref_zs = zs[:n_examples]
                ref_images = images[:n_examples]
                all_zs.append(zs[n_examples:])
                all_images.append(images[n_examples:])
            else:
                all_zs.append(model.encode(images))
                all_images.append(images)
        all_images = torch.cat(all_images, dim=0)
        all_zs = torch.cat(all_zs, dim=0)

        aa = ref_zs.sum(dim=1).unsqueeze(dim=1)
        ab = torch.matmul(ref_zs, all_zs.t())
        bb = all_zs.sum(dim=1).unsqueeze(dim=0)
        dists = torch.sqrt(aa - 2 * ab + bb)

        idxs = torch.topk(dists, k, dim=1, largest=False)[1]
        sel_images = torch.index_select(all_images, 0, idxs.view(-1))
        sel_images = sel_images.view(n_examples, k, *sel_images.images[-3:])

        ref_images = unnormalize(ref_images.cpu(), 'cifar10')
        ref_images = (ref_images.permute(0, 2, 3, 1) * 255.).numpy().astype('uint8')
        sel_images = unnormalize(sel_images.cpu(), 'cifar10')

        for i in range(n_examples):
            print(f'Image {i + 1}')
            plt.figure()
            plt.imshow(ref_images[i])
            plt.imsave(f'img_{i}.png')

            grid_img = make_grid(sel_images[i], nrow=10)
            grid_img = (grid_img.permute(1, 2, 0) * 255.).numpy().astype('uint8')

            print(f'Top {k} Nearest Neighbors (in latent space)')
            plt.figure()
            plt.imshow(grid_img)
            plt.imsave(f'nn_{i}.png')


def images_to_cuda(images):
    if isinstance(images, (tuple, list)):
        bs = images[0].shape[0]
        images = [x.cuda(non_blocking=True) for x in images]
    else:
        bs = images.shape[0]
        images = images.cuda(non_blocking=True)
    return images


model, linear_classifier, train_loader, test_loader = load_model_and_data('simclr')
evaluate_accuracy(model, linear_classifier, train_loader, test_loader)
display_nearest_neighbors(model, test_loader)
