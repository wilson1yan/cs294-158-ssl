import os.path as osp

from torchvision import datasets
from torchvision import transforms


def get_transform(task, train=True):
    transform = None
    if task == 'context_encoder':
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif task == 'rotation':
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    elif task == 'cpc':
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
    elif task == 'simclr':
        pass
    else:
        raise Exception('Invalid task:', task)

    return transform


def get_datasets(dataset, task):
    if 'imagenet' in dataset:
        train_dir = osp.join('data', dataset, 'train')
        val_dir = osp.join('data', dataset, 'val')
        train_dataset = datasets.ImageFolder(
            train_dir,
            get_transform(task, train=True)
        )

        val_dataset = datasets.ImageFolder(
            val_dir,
            get_transform(task, train=False)
        )

        return train_dataset, val_dataset, len(train_dataset.classes)
    elif dataset == 'cifar10':
        train_dset = datasets.CIFAR10(osp.join('data', dataset), train=True,
                                      transform=get_transform(task, train=True),
                                      download=True)
        test_dset = datasets.CIFAR10(osp.join('data', dataset), train=False,
                                     transform=get_transform(task, train=False),
                                     download=True)
        return train_dset, test_dset, len(train_dset.classes)
    else:
        raise Exception('Invalid dataset:', dataset)
