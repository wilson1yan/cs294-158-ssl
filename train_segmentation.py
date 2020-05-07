import argparse
import os
import os.path as osp
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler

from deepul_helper.utils import AverageMeter, ProgressMeter, remove_module_state_dict
from deepul_helper.data import get_datasets
from deepul_helper.lars import LARS
from deepul_helper.seg_model import SegmentationModel
from deepul_helper.tasks import *


parser = argparse.ArgumentParser()
# Currently only works for SimCLR
parser.add_argument('-d', '--dataset', type=str, default='pascalvoc2012',
                     help='default: pascalvoc2012')
parser.add_argument('-t', '--pretrained_dir', type=str, default='results/imagenet100_simclr',
                     help='directory of the pretrained model (default: results/imagenet100_simclr)')

# Training parameters
parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size total for all gpus (default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=200, help='default: 200')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='sgd|lars|adam (default: sgd)')
parser.add_argument('--lr', type=float, default=0.1, help='default: 0.1')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='default: 0.9')
parser.add_argument('-w', '--weight_decay', type=float, default=5e-4, help='default: 5e-4')

parser.add_argument('-p', '--port', type=int, default=23456, help='tcp port for distributed trainign (default: 23456)')
parser.add_argument('-i', '--log_interval', type=int, default=10, help='default: 10')


best_loss = float('inf')
best_map = 0.0

def main():
    args = parser.parse_args()
    assert osp.exists(args.pretrained_dir)

    ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(gpu, ngpus, args):
    global best_loss

    print(f'Starting process on GPU: {gpu}')
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                            world_size=ngpus, rank=gpu)
    total_batch_size = args.batch_size
    args.batch_size = args.batch_size // ngpus

    train_dataset, val_dataset, n_classes = get_datasets(args.dataset, 'segmentation')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=16,
        pin_memory=True, sampler=train_sampler, drop_last=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=16,
        pin_memory=True, drop_last=True, sampler=val_sampler
    )

    # Currently only supports using SimCLR
    pretrained_model = SimCLR('imagenet100', 100, dist)
    ckpt = torch.load(osp.join(args.pretrained_dir, 'model_best.pth.tar'), map_location='cpu')
    state_dict = remove_module_state_dict(ckpt['state_dict'])
    pretrained_model.load_state_dict(state_dict)
    pretrained_model = pretrained_model.cuda(gpu)
    pretrained_model.eval()
    print(f"Loaded pretrained model at Epoch {ckpt['epoch']} Acc {ckpt['best_acc']:.2f}")

    model = SegmentationModel(n_classes)

    args.metrics = model.metrics
    args.metrics_fmt = model.metrics_fmt

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    args.gpu = gpu

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    else:
        raise Exception('Unsupported optimizer', args.optimizer)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0, -1)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        train(train_loader, pretrained_model, model, optimizer, epoch, args)

        val_loss, val_acc, val_miou = validate(val_loader, pretrained_model, model, args, dist)

        scheduler.step()

        if dist.get_rank() == 0:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_acc': val_acc,
                'best_miou': val_miou
            }, is_best, args)


def train(train_loader, pretrained_model, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('PixelAcc@1', ':6.2f')
    top3 = AverageMeter('PixelAcc@3', ':6.2f')
    miou = AverageMeter('mIOU', ':6.2f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(args.metrics, args.metrics_fmt)}
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, top1, top3, miou] + list(avg_meters.values()),
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute loss
        bs = images.shape[0]
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            features = pretrained_model.get_features(images)

        out, logits = model(features, target)
        for k, v in out.items():
            avg_meters[k].update(v.item(), bs)

        # compute gradient and optimizer step for ssl task
        optimizer.zero_grad()
        out['Loss'].backward()
        optimizer.step()

        miou.update(compute_mIOU(logits, target), bs)
        acc1, acc3 = accuracy(logits, target, topk=(1, 3))
        top1.update(acc1[0], bs)
        top3.update(acc3[0], bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            progress.display(i)


def validate(val_loader, pretrained_model, model, args, dist):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('PixelAcc@1', ':6.2f')
    top3 = AverageMeter('PixelAcc@3', ':6.2f')
    miou = AverageMeter('mIOU', ':6.2f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(args.metrics, args.metrics_fmt)}
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, top1, top3, miou] + list(avg_meters.values()),
        prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute and measure loss
            bs = images.shape[0]
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            features = pretrained_model.get_features(images)
            out, logits = model(features, target)
            for k, v in out.items():
                avg_meters[k].update(v.item(), bs)

            miou.update(compute_mIOU(logits, target), bs)
            acc1, acc3 = accuracy(logits, target, topk=(1, 3))
            top1.update(acc1[0], bs)
            top3.update(acc3[0], bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                progress.display(i)

    data = torch.FloatTensor([avg_meters['Loss'].avg, top1.avg, top3.avg, miou.avg] + \
                             [v.avg for v in avg_meters.values()])
    data = data.cuda(args.gpu)
    gather_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, data)
    data = torch.stack(gather_list, dim=0).mean(0).cpu().numpy()

    if dist.get_rank() == 0:
        print_str = f' * PixelAcc@1 {data[1]:.3f} PixelAcc@3 {data[2]:.3f} mIOU {data[3]:.3f}'
        for i, (k, v) in enumerate(avg_meters.items()):
            print_str += f' {k} {data[i+3]:.3f}'
        print(print_str)

    dist.barrier()
    return data[0], data[1], data[3]


def save_checkpoint(state, is_best, args, filename='seg_checkpoint.pth.tar'):
    filename = osp.join(args.pretrained_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(args.pretrained_dir, 'seg_model_best.pth.tar'))


def accuracy(logits, target, topk=(1,)):
    # Assumes logits (B, n_classes, H, W), target (B, H, W)
    B, n_classes, H, W = logits.shape
    logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, n_classes)
    target = target.view(-1)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / (batch_size * H * W)))
        return res

def compute_mIOU(logits, target):
    # Assumes logits (B, n_classes, H, W), target (B, H, W)
    n_classes = logits.shape[1]
    pred = torch.argmax(logits, dim=1)

    intersection = pred * (pred == target)
    area_intersection = torch.histc(intersection, bins=n_classes, min=0, max=n_classes-1)

    area_pred = torch.histc(pred, bins=n_classes, min=0, max=n_classes - 1)
    area_target = torch.histc(target, bins=n_classes, min=0, max=n_classes - 1)
    area_union = area_pred + area_target - area_intersection

    return torch.mean(area_intersection / (area_union + 1e-10)) * 100.


if __name__ == '__main__':
    main()
