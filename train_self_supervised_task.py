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

from deepul_helper.models import ContextEncoder, RotationPrediction, CPCModel, SimCLR
from deepul_helper.utils import AverageMeter, ProgressMeter
from deepul_helper.data import get_datasets


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='imagenet100')
parser.add_argument('-t', '--task', type=str, required=True,
                    help='self-supervised learning task (context_encoder|rotation|cpc|simclr)')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size total for all gpus')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-i', '--log_interval', type=int, default=10)


best_loss = float('inf')
best_acc = 0.0

def main():
    args = parser.parse_args()
    assert args.task in ['context_encoder', 'rotation', 'cpc', 'simclr']

    args.output_dir = osp.join('results', f"{args.dataset}_{args.task}")
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))


def main_worker(gpu, ngpus, args):
    global best_loss

    print(f'Starting process on GPU: {gpu}')
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456',
                            world_size=ngpus, rank=gpu)
    args.batch_size = args.batch_size // ngpus

    if args.task == 'context_encoder':
        model = ContextEncoder()
    elif args.task == 'rotation':
        model = RotationPrediction()
    elif args.task == 'cpc':
        model = CPCModel()
    elif args.task == 'simclr':
        model = SimCLR()
    else:
        raise Exception('Invalid task:', args.task)
    args.metrics = model.metrics
    args.metrics_fmt = model.metrics_fmt

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    args.gpu = gpu

    train_dataset, val_dataset, n_classes = get_datasets(args.dataset, args.task)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True
    )

    linear_classifier = nn.Sequential(nn.BatchNorm1d(model.latent_dim),
                                      nn.Linear(model.latent_dim, n_classes)).cuda()
    linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[gpu])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer_linear = torch.optim.Adam(linear_classifier.parameters(), args.lr)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, linear_classifier,
              optimizer, optimizer_linear, epoch, args)

        val_loss = validate(val_loader, model, linear_classifier, args)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if gpu == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, args)


def train(train_loader, model, linear_classifier, optimizer,
          optimizer_linear, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('LinearAcc@1', ':6.2f')
    top5 = AverageMeter('LinearAcc@5', ':6.2f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(args.metrics, args.metrics_fmt)}
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, top1, top5] + list(avg_meters.values()),
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()
    linear_classifier.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute loss
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        out, zs = model(images)
        for k, v in out.items():
            avg_meters[k].update(v.item(), images.shape[0])

        # compute gradient and optimizer step for ssl task
        optimizer.zero_grad()
        out['Loss'].backward()
        optimizer.step()

        # compute gradient and optimizer step for classifier
        logits = linear_classifier(zs.detach())
        loss = F.cross_entropy(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1.update(acc1[0], images.shape[0])
        top5.update(acc5[0], images.shape[0])

        optimizer_linear.zero_grad()
        loss.backward()
        optimizer_linear.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            progress.display(i)


def validate(val_loader, model, linear_classifier, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('LinearAcc@1', ':6.2f')
    top5 = AverageMeter('LinearAcc@5', ':6.2f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(args.metrics, args.metrics_fmt)}
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, top1, top5] + list(avg_meters.values()),
        prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute and measure loss
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            out, zs = model(images)
            for k, v in out.items():
                avg_meters[k].update(v.item(), images.shape[0])

            logits = linear_classifier(zs)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1.update(acc1[0], images.shape[0])
            top5.update(acc5[0], images.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                progress.display(i)

    print_str = f' * LinearAcc@1 {top1.avg:.3f} LinearAcc@5 {top5.avg:.3f}'
    for k, v in avg_meters.items():
        print_str += f' {k} {v.avg:.3f}'
    print(print_str)

    return avg_meters['Loss'].avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = osp.join(args.output_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(args.output_dir, 'model_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
