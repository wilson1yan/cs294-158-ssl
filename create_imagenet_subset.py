import shutil
import sys
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

from torchvision.datasets import ImageFolder


n_classes = int(sys.argv[1])
print('Creating a subset of ImageNet with {} classes'.format(n_classes))

dset_dir = osp.join('data', 'imagenet')
dset = ImageFolder(osp.join(dset_dir, 'train'))
classes = dset.classes

new_dset_dir = osp.join('data', 'imagenet{}'.format(n_classes))
classes_subset = np.random.choice(classes, size=n_classes, replace=False)

os.makedirs(osp.join(new_dset_dir, 'train'))
os.makedirs(osp.join(new_dset_dir, 'val'))

for c in tqdm(classes_subset):
    src = osp.join(dset_dir, 'train', c)
    dst = osp.join(new_dset_dir, 'train', c)
    shutil.copytree(src, dst)

    src = osp.join(dset_dir, 'val', c)
    dst = osp.join(new_dset_dir, 'val', c)
    shutil.copytree(src, dst)
