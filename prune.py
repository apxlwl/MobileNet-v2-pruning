from pruner import l1normPruner
import pruner
import os
import argparse
import torch
from torchvision import datasets, transforms
from models import *
import torch.optim as optim
from os.path import join
import json

from thop import clever_format, profile

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Mobilev2 Pruner')
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
parser.add_argument('--finetunelr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='checkpoints/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='checkpoints', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='MobileNetV2', type=str, choices=['USMobileNetV2', 'MobileNetV2','VGG',
                                                                        'ShuffleNetV2','resnet50'],
                    help='architecture to use')
parser.add_argument('--pruner', default='SlimmingPruner', type=str,
                    choices=['AutoSlimPruner', 'SlimmingPruner', 'l1normPruner'],
                    help='architecture to use')
parser.add_argument('--pruneratio', default=0.4, type=float,
                    help='architecture to use')
parser.add_argument('--sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
savepath = os.path.join(args.save, args.arch, 'sr' if args.sr else 'nosr')
args.savepath = savepath
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
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

model = eval(args.arch)(input_size=32)
newmodel = eval(args.arch)(input_size=32)
if args.arch == 'USMobileNetV2':
    model.load_state_dict(torch.load(join(savepath, 'trans.pth')))
else:
    model.load_state_dict(torch.load(join(savepath, 'model_best.pth.tar'))['state_dict'])

print("Best trained model loaded.")


if args.cuda:
    model.cuda().eval()
    newmodel.cuda().eval()
best_prec1 = -1
optimizer = optim.SGD(model.parameters(), lr=args.finetunelr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.pruner == 'l1normPruner':
    kwargs = {'pruneratio': args.pruneratio}
elif args.pruner == 'SlimmingPruner':
    kwargs = {'pruneratio': args.pruneratio}
elif args.pruner == 'AutoSlimPruner':
    kwargs = {'prunestep': 16, 'constrain': 200e6}

pruner = pruner.__dict__[args.pruner](model=model, newmodel=newmodel, testset=test_loader, trainset=train_loader,
                                      optimizer=optimizer, args=args, **kwargs)
pruner.prune()
##---------count op
input = torch.randn(1, 3, 32, 32).cuda()
flops, params = profile(model, inputs=(input,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
flopsnew, paramsnew = profile(newmodel, inputs=(input,), verbose=False)
flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")
print("flops:{}->{}, params: {}->{}".format(flops, flopsnew, params, paramsnew))
accold = pruner.test(newmodel=False, cal_bn=False)
accpruned = pruner.test(newmodel=True)
accfinetune = pruner.finetune()

print("original performance:{}, pruned performance:{},finetuned:{}".format(accold, accpruned, accfinetune))

with open(join(savepath, '{}.json'.format(args.pruneratio)), 'w') as f:
    json.dump({
        'accuracy_original': accold,
        'accuracy_pruned': accpruned,
        'accuracy_finetune': accfinetune,
    }, f)
