import argparse
import os
import os.path as osp
import time
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from deepul_helper.models import CPCModel


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size PER GPU')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-i', '--log_interval', type=int, default=10)
parser.add_argument('-o', '--output_dir', type=str, default='cpc')


best_loss = float('inf')

def main():
    args = parser.parse_args()
    args.dataset = osp.join('data', args.dataset)
    args.output_dir = osp.join('results', args.output_dir)
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))


def main_worker(gpu, ngpus, args):
    global best_loss

    print(f'Starting process on GPU: {gpu}')
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456',
                            world_size=ngpus, rank=gpu)

    model = CPCModel()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    args.gpu = gpu
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_dir = osp.join(args.dataset, 'train')
    train_dataset = ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, sampler=train_sampler
    )

    val_dir = osp.join(args.dataset, 'val')
    val_dataset = ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(256),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True
    )

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, optimizer, epoch, args)

        val_loss = validate(val_loader, model, args)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if gpu == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, args)


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute loss
        images = images.cuda(args.gpu, non_blocking=True)
        loss = model(images)
        losses.update(loss.item(), images.shape[0])

        # compute gradient and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            progress.display(i)


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses],
        prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, _) in enumerate(val_loader):
            # compute and measure loss
            images = images.cuda(args.gpu, non_blocking=True)
            loss = model(images)
            losses.update(loss.item(), images.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                progress.display(i)

    return losses.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = osp.join(args.output_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(args.output_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
