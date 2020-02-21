from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import *
from tqdm import tqdm
from models.slimmableops import bn_calibration_init
import USconfig as FLAGS
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='checkpoints', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='USMobileNetV2', type=str,choices=['MobileNetV2','USMobileNetV2'],
                    help='architecture to use')
parser.add_argument('--sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--test',action='store_true')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)
savepath = os.path.join(args.save, args.arch, 'sr' if args.sr else 'nosr')

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = eval(args.arch)(input_size=32)
if args.cuda:
    model.cuda()
best_prec1 = -1
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1


def train():
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
def trainUS():
    max_width = max(FLAGS.width_mult_list)
    min_width = min(FLAGS.width_mult_list)

    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        ###
        widths_train = []
        for _ in range(getattr(FLAGS, 'num_sample_training', 2) - 2):
            widths_train.append(
                random.uniform(min_width, max_width))
        widths_train = [max_width, min_width] + widths_train
        # widths_train = [min_width]
        for width_mult in widths_train:
            # TODO :add inplace distillation
            model.apply(lambda m: setattr(
                m, 'width_mult',
                width_mult))
                # always track largest model and smallest model
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
        ###
        optimizer.step()

def test(test_width=1.0,recal=False):
    model.eval()
    test_loss = 0
    correct = 0
    model.apply(lambda m: setattr(m, 'width_mult',test_width))
    if recal:
        model.apply(bn_calibration_init)
        model.train()
        for idx,(data, target) in enumerate(tqdm(train_loader, total=len(train_loader))):
            if idx==FLAGS.recal_batch:
                break
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            del output
    model.eval()
    for data, target in tqdm(test_loader, total=len(test_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct.item() / float(len(test_loader.dataset))
def export2normal():
    newmodel=MobileNetV2()
    from collections import OrderedDict
    statedic=[]
    for k2,v in model.state_dict().items():
        if 'running' in k2 or 'num_batches_tracked' in k2:
            continue
        statedic.append(v)
    names=[]
    for k1,v1 in newmodel.state_dict().items():
        if 'running' in k1 or 'num_batches_tracked' in k1:
            continue
        names.append(k1)
    newdic=OrderedDict(zip(names,statedic))
    newmodel.load_state_dict(newdic,strict=False)
    torch.save(newmodel.state_dict(),os.path.join(savepath,'trans.pth'))
    print("save transferred ckpt at {}".format(os.path.join(savepath,'trans.pth')))

best_prec1 = 0. if best_prec1 == -1 else best_prec1
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=0)
if args.test:
    if args.arch=='USMobileNetV2':
        export2normal()
        res_acc=[1.0]*len(FLAGS.width_mult_list)
        for idx,width in enumerate(FLAGS.width_mult_list):
            acc=test(width,recal=True)
            res_acc[idx]=acc
            print("Test accuracy for width {} is {}".format(width,acc))
    else:
        print("Test accuracy {}".format(test()))
else:
    for epoch in range(args.start_epoch, args.epochs):
        if args.arch=='USMobileNetV2':
            trainUS()
            prec1=test(test_width=1.0,recal=False)
        else:
            train()
            prec1 = test()
        scheduler.step(epoch)
        lr_current = optimizer.param_groups[0]['lr']
        print("currnt lr:{}".format(lr_current))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            ckptfile = os.path.join(savepath, 'model_best.pth.tar')
        else:
            ckptfile = os.path.join(savepath, 'checkpoint.pth.tar')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, ckptfile)
