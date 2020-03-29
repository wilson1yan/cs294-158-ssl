import os.path as osp

from torchvision import datasets
from torchvision import transforms


def get_datasets(dataset):
    if dataset == 'imagenet':
        train_dir = osp.join('data', dataset, 'train')
        val_dir = osp.join('data', dataset, 'val')
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )

        val_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )

        return train_dataset, val_dataset, len(train_dataset.classes)
    elif dataset == 'cifar10':
        root = osp.join('data', dataset)
        train_dataset = datasets.CIFAR10(root, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Grayscale(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))
                                         ]))
        test_dataset = datasets.CIFAR10(root, train=False, download=True,
                                        transforms=transforms.Compose([
                                            transforms.Grayscale(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))
                                        ]))
        return train_dataset, test_dataset, len(train_dataset.classes)
    else:
        raise Exception('Invalid dataset:', dataset)
