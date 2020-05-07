import os.path as osp
import random

import numpy as np
import cv2

import torchvision.transforms.functional as F
from torchvision import datasets
from torchvision import transforms


def get_transform(dataset, task, train=True):
    transform = None
    if task == 'context_encoder':
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        elif 'imagenet' in dataset:
            transform = transforms.Compose([
                transforms.Resize(350),
                transforms.RandomCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    elif task == 'rotation':
        if dataset == 'cifar10':
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        elif 'imagenet' in dataset:
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
    elif task == 'cpc':
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    elif task == 'simclr':
        if dataset == 'cifar10':
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
        elif 'imagenet' in dataset:
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=11),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(128),
                    transforms.CenterCrop(128),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        transform = SimCLRDataTransform(transform)
    elif task == 'segmentation':
        if train:
            transform = MultipleCompose([
                MultipleRandomResizedCrop(128),
                MultipleRandomHorizontalFlip(),
                RepeatTransform(transforms.ToTensor()),
                GroupTransform([
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    SegTargetTransform()])
            ])
        else:
            transform = MultipleCompose([
                RepeatTransform(transforms.Resize(128)),
                RepeatTransform(transforms.CenterCrop(128)),
                RepeatTransform(transforms.ToTensor()),
                GroupTransform([
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    SegTargetTransform()])
            ])
    else:
        raise Exception('Invalid task:', task)

    return transform


def get_datasets(dataset, task):
    if 'imagenet' in dataset:
        train_dir = osp.join('data', dataset, 'train')
        val_dir = osp.join('data', dataset, 'val')
        train_dataset = datasets.ImageFolder(
            train_dir,
            get_transform(dataset, task, train=True)
        )

        val_dataset = datasets.ImageFolder(
            val_dir,
            get_transform(dataset, task, train=False)
        )

        return train_dataset, val_dataset, len(train_dataset.classes)
    elif dataset == 'cifar10':
        train_dset = datasets.CIFAR10(osp.join('data', dataset), train=True,
                                      transform=get_transform(dataset, task, train=True),
                                      download=True)
        test_dset = datasets.CIFAR10(osp.join('data', dataset), train=False,
                                     transform=get_transform(dataset, task, train=False),
                                     download=True)
        return train_dset, test_dset, len(train_dset.classes)
    elif dataset == 'pascalvoc2012':
        train_dset = datasets.VOCSegmentation(osp.join('data', dataset), image_set='train',
                                              transforms=get_transform(dataset, task, train=True),
                                              download=True)
        test_dset = datasets.VOCSegmentation(osp.join('data', dataset), image_set='val',
                                             transforms=get_transform(dataset, task, train=False),
                                             download=True)
        return train_dset, test_dset, 21
    else:
        raise Exception('Invalid dataset:', dataset)


# https://github.com/sthalles/SimCLR/blob/master/data_aug/gaussian_blur.py
class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

# Re-written torchvision transforms to support operations on multiple inputs
# Needed to maintain consistency on random transforms with real images and their segmentations
class MultipleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        for t in self.transforms:
            inputs = t(*inputs)
        return inputs


class GroupTransform(object):
    """ Applies a list of transforms elementwise """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        assert len(inputs) == len(self.transforms)
        outputs = [t(inp) for t, inp in zip(self.transforms, inputs)]
        return outputs

class MultipleRandomResizedCrop(transforms.RandomResizedCrop):

    def __call__(self, *imgs):
        """
        Args:
            imgs (List of PIL Image): Images to be cropped and resized.
                                      Assumes they are all the same size

        Returns:
            PIL Images: Randomly cropped and resized images.
        """
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
                for img in imgs]

class MultipleRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, *imgs):
        if random.random() < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs

class RepeatTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, *inputs):
        return [self.transform(inp) for inp in inputs]

class SegTargetTransform(object):
    def __call__(self, target):
        target *= 255.
        target[target > 20] = 0
        return target.long()
