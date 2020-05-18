import argparse
import os
import os.path as osp
import time
import shutil
from warmup_scheduler import GradualWarmupScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler

from deepul_helper.tasks import *
from deepul_helper.utils import AverageMeter, ProgressMeter, accuracy
from deepul_helper.data import get_datasets
from deepul_helper.lars import LARS


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='cifar10|imagenet* (default: cifar10)')
parser.add_argument('-t', '--task', type=str, default='rotation',
                    help='context_encoder|rotation|cpc|simclr (default: rotation)')

# Training parameters
parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size total for all gpus (default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=200, help='default: 200')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='sgd|lars|adam (default: sgd)')
parser.add_argument('--lr', type=float, default=0.1, help='default: 0.1')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='default: 0.9')
parser.add_argument('-w', '--weight_decay', type=float, default=5e-4, help='default: 5e-4')
parser.add_argument('-u', '--warmup_epochs', type=int, default=0,
                    help='# of warmup epochs. If > 0, then the scheduler warmups from lr * batch_size / 256.')

parser.add_argument('-p', '--port', type=int, default=23456, help='tcp port for distributed trainign (default: 23456)')
parser.add_argument('-i', '--log_interval', type=int, default=10, help='default: 10')


best_loss = float('inf')
best_acc = 0.0

def main():
    args = parser.parse_args()
    assert args.task in ['context_encoder', 'rotation', 'cpc', 'simclr']

    args.output_dir = osp.join('results', f"{args.dataset}_{args.task}")
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(gpu, ngpus, args):
    global best_loss

    print(f'Starting process on GPU: {gpu}')
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                            world_size=ngpus, rank=gpu)
    total_batch_size = args.batch_size
    args.batch_size = args.batch_size // ngpus

    train_dataset, val_dataset, n_classes = get_datasets(args.dataset, args.task)
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

    if args.task == 'context_encoder':
        model = ContextEncoder(args.dataset, n_classes)
    elif args.task == 'rotation':
        model = RotationPrediction(args.dataset, n_classes)
    elif args.task == 'cpc':
        model = CPC(args.dataset, n_classes)
    elif args.task == 'simclr':
        model = SimCLR(args.dataset, n_classes, dist)
    else:
        raise Exception('Invalid task:', args.task)
    args.metrics = model.metrics
    args.metrics_fmt = model.metrics_fmt

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    args.gpu = gpu

    linear_classifier = model.construct_classifier().cuda(gpu)
    linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[gpu], find_unused_parameters=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
        optimizer_linear = torch.optim.SGD(linear_classifier.parameters(), lr=args.lr,
                                           momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)
        optimizer_linear = torch.optim.Adam(linear_classifier.parameters(), lr=args.lr,
                                            betas=(args.momentum, 0.999))
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
        optimizer_linear = LARS(linear_classifier.parameters(), lr=args.lr,
                                momentum=args.momentum)
    else:
        raise Exception('Unsupported optimizer', args.optimizer)

    # Minimize SSL task loss, maximize linear classification accuracy
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0, -1)
    scheduler_linear = lr_scheduler.CosineAnnealingLR(optimizer_linear, args.epochs, 0, -1)
    if args.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=total_batch_size / 256.,
                                           total_epoch=args.warmup_epochs, after_scheduler=scheduler)
        scheduler_linear = GradualWarmupScheduler(optimizer, multiplier=total_batch_size / 256.,
                                                  total_epoch=args.warmup_epochs,
                                                  after_scheduler=scheduler_linear)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, linear_classifier,
              optimizer, optimizer_linear, epoch, args)

        val_loss, val_acc = validate(val_loader, model, linear_classifier, args, dist)

        scheduler.step()
        scheduler_linear.step()

        if dist.get_rank() == 0:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'state_dict_linear': linear_classifier.state_dict(),
                'optimizer_linear': optimizer_linear.state_dict(),
                'schedular_linear': scheduler_linear.state_dict(),
                'best_loss': best_loss,
                'best_acc': val_acc
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
        if isinstance(images, (tuple, list)):
            # Special case for SimCLR which returns a tuple of 2 image batches
            bs = images[0].shape[0]
            images = [x.cuda(args.gpu, non_blocking=True)
                      for x in images]
        else:
            bs = images.shape[0]
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        out, zs = model(images)
        zs = zs.detach()
        for k, v in out.items():
            avg_meters[k].update(v.item(), bs)

        # compute gradient and optimizer step for ssl task
        optimizer.zero_grad()
        out['Loss'].backward()
        optimizer.step()

        # compute gradient and optimizer step for classifier
        logits = linear_classifier(zs)
        loss = F.cross_entropy(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1.update(acc1[0], bs)
        top5.update(acc5[0], bs)

        optimizer_linear.zero_grad()
        loss.backward()
        optimizer_linear.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            progress.display(i)


def validate(val_loader, model, linear_classifier, args, dist):
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
    linear_classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute and measure loss
            if isinstance(images, (tuple, list)):
                # Special case for SimCLR which returns a tuple of 2 image batches
                bs = images[0].shape[0]
                images = [x.cuda(args.gpu, non_blocking=True)
                        for x in images]
            else:
                bs = images.shape[0]
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            out, zs = model(images)
            for k, v in out.items():
                avg_meters[k].update(v.item(), bs)

            logits = linear_classifier(zs)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1.update(acc1[0], bs)
            top5.update(acc5[0], bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                progress.display(i)

    data = torch.FloatTensor([avg_meters['Loss'].avg, top1.avg, top5.avg] + [v.avg for v in avg_meters.values()])
    data = data.cuda(args.gpu)
    gather_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, data)
    data = torch.stack(gather_list, dim=0).mean(0).cpu().numpy()

    if dist.get_rank() == 0:
        print_str = f' * LinearAcc@1 {data[1]:.3f} LinearAcc@5 {data[2]:.3f}'
        for i, (k, v) in enumerate(avg_meters.items()):
            print_str += f' {k} {data[i+3]:.3f}'
        print(print_str)

    dist.barrier()
    return data[0], data[1]


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = osp.join(args.output_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(args.output_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
