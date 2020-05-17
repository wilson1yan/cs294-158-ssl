import argparse
import os
import os.path as osp
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import save_image

from deepul_helper.utils import AverageMeter, ProgressMeter, remove_module_state_dict, seg_idxs_to_color
from deepul_helper.data import get_datasets
from deepul_helper.seg_model import SegmentationModel
from deepul_helper.tasks import *


parser = argparse.ArgumentParser()
# Currently only works for SimCLR
parser.add_argument('-d', '--dataset', type=str, default='pascalvoc2012',
                     help='default: pascalvoc2012')
parser.add_argument('-t', '--pretrained_dir', type=str, default='results/imagenet100_simclr',
                     help='directory of the pretrained model (default: results/imagenet100_simclr)')

# Training parameters
parser.add_argument('-b', '--batch_size', type=int, default=16, help='default: 128')
parser.add_argument('-e', '--epochs', type=int, default=1000, help='default: 200')
parser.add_argument('-o', '--optimizer', type=str, default='adam', help='sgd|adam (default: adam)')
parser.add_argument('--lr', type=float, default=1e-4, help='default: 1e-3')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='default: 0.9')
parser.add_argument('-w', '--weight_decay', type=float, default=5e-4, help='default: 5e-4')
parser.add_argument('-i', '--log_interval', type=int, default=10, help='default: 10')
parser.add_argument('-f', '--fine_tuning', action='store_true', help='fine-tune the pretrained model')

best_loss = float('inf')

def main():
    global best_loss

    args = parser.parse_args()
    assert osp.exists(args.pretrained_dir)

    args.seg_dir = osp.join(args.pretrained_dir, 'segmentation')
    if not osp.exists(args.seg_dir):
        os.makedirs(args.seg_dir)

    train_dataset, val_dataset, n_classes = get_datasets(args.dataset, 'segmentation')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=16,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=16,
        pin_memory=True
    )

    # Currently only supports using SimCLR
    pretrained_model = SimCLR('imagenet100', 100, None)
    ckpt = torch.load(osp.join(args.pretrained_dir, 'model_best.pth.tar'), map_location='cpu')
    state_dict = remove_module_state_dict(ckpt['state_dict'])
    pretrained_model.load_state_dict(state_dict)
    pretrained_model.cuda()
    if not args.fine_tuning:
        pretrained_model.eval()
    print(f"Loaded pretrained model at Epoch {ckpt['epoch']} Acc {ckpt['best_acc']:.2f}")

    model = SegmentationModel(n_classes)

    args.metrics = model.metrics
    args.metrics_fmt = model.metrics_fmt

    torch.backends.cudnn.benchmark = True
    model.cuda()

    params = list(model.parameters())
    if args.fine_tuning:
        params += list(pretrained_model.parameters())
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)
    else:
        raise Exception('Unsupported optimizer', args.optimizer)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0, -1)

    for epoch in range(args.epochs):
        train(train_loader, pretrained_model, model, optimizer, epoch, args)
        val_loss, val_acc, val_miou = validate(val_loader, pretrained_model, model, args, dist)

        scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'pt_state_dict': pretrained_model.state_dict(),
            'best_loss': best_loss,
            'best_acc': val_acc,
            'best_miou': val_miou
        }, is_best, args)

        # Save segmentation samples to visualize
        if epoch % 10 == 0:
            with torch.no_grad():
                images, target = next(iter(val_loader))
                images, target = images[:33], target[:33]
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True).long().squeeze(1)
                features = pretrained_model.get_features(images)
                _, logits = model(features, target)
                pred = torch.argmax(logits, dim=1)

                target = seg_idxs_to_color(target.cpu())
                pred = seg_idxs_to_color(pred.cpu())
                images = unnormalize(images.cpu())

                to_save = torch.stack((images, target, pred), dim=1).flatten(end_dim=1)
                save_image(to_save, osp.join(args.seg_dir, f'epoch{epoch}.png'), nrow=10, pad_value=1.)


def unnormalize(images):
    mu = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    stddev = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return images * stddev + mu


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
    if args.fine_tuning:
        pretrained_model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute loss
        bs = images.shape[0]
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).squeeze(1).long()

        features = pretrained_model.get_features(images)
        if not args.fine_tuning:
            features = [f.detach() for f in features]

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
    pretrained_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute and measure loss
            bs = images.shape[0]
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).squeeze(1).long()

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

    print_str = f' * PixelAcc@1 {top1.avg:.3f} PixelAcc@3 {top3.avg:.3f} mIOU {miou.avg:.3f}'
    for k, v in avg_meters.items():
        print_str += f' {k} {v.avg:.3f}'
    print(print_str)

    return avg_meters['Loss'].avg, top1.avg, miou.avg


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
            res.append(correct_k.mul_(100.0 / (B * H * W)))
        return res

def compute_mIOU(logits, target):
    # Assumes logits (B, n_classes, H, W), target (B, H, W)
    n_classes = logits.shape[1]
    pred = torch.argmax(logits, dim=1)

    # Ignore background class 0
    intersection = pred * (pred == target)
    area_intersection = torch.histc(intersection, bins=n_classes - 1, min=1, max=n_classes-1)

    area_pred = torch.histc(pred, bins=n_classes - 1, min=1, max=n_classes - 1)
    area_target = torch.histc(target, bins=n_classes - 1, min=1, max=n_classes - 1)
    area_union = area_pred + area_target - area_intersection

    return torch.mean(area_intersection / (area_union + 1e-10)) * 100.


if __name__ == '__main__':
    main()
