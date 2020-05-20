import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision.utils import make_grid

from deepul_helper.data import get_datasets
from deepul_helper.tasks import *
from deepul_helper.utils import accuracy, unnormalize, remove_module_state_dict, seg_idxs_to_color
from deepul_helper.seg_model import SegmentationModel


def load_model_and_data(task, dataset='cifar10'):
    train_dset, test_dset, n_classes = get_datasets(dataset, task)
    train_loader = data.DataLoader(train_dset, batch_size=128, num_workers=4,
                                   pin_memory=True, shuffle=True)
    test_loader = data.DataLoader(test_dset, batch_size=128, num_workers=4,
                                  pin_memory=True, shuffle=True)

    ckpt_pth = osp.join('results', f'{dataset}_{task}', 'model_best.pth.tar')
    ckpt = torch.load(ckpt_pth, map_location='cpu')

    if task == 'context_encoder':
        model = ContextEncoder(dataset, n_classes)
    elif task == 'rotation':
        model = RotationPrediction(dataset, n_classes)
    elif task == 'simclr':
        model = SimCLR(dataset, n_classes, None)
    model.load_state_dict(remove_module_state_dict(ckpt['state_dict']))

    model.cuda()
    model.eval()

    linear_classifier = model.construct_classifier()
    linear_classifier.load_state_dict(remove_module_state_dict(ckpt['state_dict_linear']))

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

            correct1 += acc1.item() * logits.shape[0]
            correct5 += acc5.item() * logits.shape[0]
    total = len(loader.dataset)

    return correct1 / total, correct5 / total


def display_nearest_neighbors(task, model, loader, n_examples=4, k=16):
    with torch.no_grad():
        all_images, all_zs = [], []
        for i, (images, _) in enumerate(loader):
            images = images_to_cuda(images)
            if task == 'simclr':
                images = images[0]
            zs = model.encode(images)

            images = images.cpu()
            zs = zs.cpu()

            if i == 0:
                ref_zs = zs[:n_examples]
                ref_images = images[:n_examples]
                all_zs.append(zs[n_examples:])
                all_images.append(images[n_examples:])
            else:
                all_zs.append(zs)
                all_images.append(images)
        all_images = torch.cat(all_images, dim=0)
        all_zs = torch.cat(all_zs, dim=0)

        aa = (ref_zs ** 2).sum(dim=1).unsqueeze(dim=1)
        ab = torch.matmul(ref_zs, all_zs.t())
        bb = (all_zs ** 2).sum(dim=1).unsqueeze(dim=0)
        dists = torch.sqrt(aa - 2 * ab + bb)

        idxs = torch.topk(dists, k, dim=1, largest=False)[1]
        sel_images = torch.index_select(all_images, 0, idxs.view(-1))
        sel_images = unnormalize(sel_images.cpu(), 'cifar10')
        sel_images = sel_images.view(n_examples, k, *sel_images.shape[-3:])

        ref_images = unnormalize(ref_images.cpu(), 'cifar10')
        ref_images = (ref_images.permute(0, 2, 3, 1) * 255.).numpy().astype('uint8')

        for i in range(n_examples):
            print(f'Image {i + 1}')
            plt.figure()
            plt.axis('off')
            plt.imshow(ref_images[i])
            plt.show()

            grid_img = make_grid(sel_images[i], nrow=4)
            grid_img = (grid_img.permute(1, 2, 0) * 255.).numpy().astype('uint8')

            print(f'Top {k} Nearest Neighbors (in latent space)')
            plt.figure()
            plt.axis('off')
            plt.imshow(grid_img)
            plt.show()


def images_to_cuda(images):
    if isinstance(images, (tuple, list)):
        images = [x.cuda(non_blocking=True) for x in images]
    else:
        images = images.cuda(non_blocking=True)
    return images


def show_context_encoder_inpainting():
    model, _, _, test_loader = load_model_and_data('context_encoder', 'cifar10')
    images = next(iter(test_loader))[0][:8]
    with torch.no_grad():
        images = images.cuda(non_blocking=True)
        images_masked, images_recon = model.reconstruct(images)
        images_masked = unnormalize(images_masked.cpu(), 'cifar10')
        images_recon = unnormalize(images_recon.cpu(), 'cifar10')

        images = torch.stack((images_masked, images_recon), dim=1).flatten(end_dim=1)

        grid_img = make_grid(images, nrow=4)
        grid_img = (grid_img.permute(1, 2, 0) * 255.).numpy().astype('uint8')

        plt.figure()
        plt.axis('off')
        plt.imshow(grid_img)
        plt.show()


def show_segmentation():
    _, val_dset, n_classes = get_datasets('pascalvoc2012', 'segmentation')
    val_loader = data.DataLoader(val_dset, batch_size=128)

    pretrained_model = SimCLR('imagenet100', 100, None)
    ckpt = torch.load(osp.join('results', 'imagenet100_simclr', 'seg_model_best.pth.tar'),
                      map_location='cpu')
    pretrained_model.load_state_dict(ckpt['pt_state_dict'])
    pretrained_model.cuda().eval()

    seg_model = SegmentationModel(n_classes)
    seg_model.load_state_dict(ckpt['state_dict'])
    seg_model.cuda().eval()

    images, target = next(iter(val_loader))
    images, target = images[:12], target[:12]
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True).long().squeeze(1)
    features = pretrained_model.get_features(images)
    _, logits = seg_model(features, target)
    pred = torch.argmax(logits, dim=1)

    target = seg_idxs_to_color(target.cpu(), 'palette.pkl')
    pred = seg_idxs_to_color(pred.cpu(), 'palette.pkl')
    images = unnormalize(images.cpu(), 'imagenet')

    to_show = torch.stack((images, target, pred), dim=1).flatten(end_dim=1)
    to_show = make_grid(to_show, nrow=6, pad_value=1.)
    to_show = (to_show.permute(1, 2, 0) * 255.).numpy().astype('uint8')

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(to_show)
    plt.show()

