import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from models import MobileNetV2, InvertedResidual, sepconv_bn, conv_bn_relu
from pruner.Block import *

from models.vgg import conv_bn_relu

class BasePruner:
    def __init__(self, model, newmodel, testset, trainset, optimizer,args):
        self.model = model
        self.newmodel = newmodel
        self.testset = testset
        self.trainset = trainset
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3,threshold=1e-2)
        self.args=args
        self.blocks=[]
    def prune(self):
        self.blocks = []
        for midx, (name, module) in enumerate(self.model.named_modules()):
            idx = len(self.blocks)
            if isinstance(module, InvertedResidual):
                self.blocks.append(InverRes(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
            if isinstance(module, conv_bn_relu):
                print(module)
                for k,v in module.state_dict().items():
                    print(k,v.shape)
                self.blocks.append(CB(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
            if isinstance(module, nn.Linear):
                self.blocks.append(FC(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
        # special blocks
        for b in self.blocks:
            if b.layername=='features.18':
                b.keepoutput=True
                b.bnscale=None
    def test(self, newmodel=True,ckpt=None,cal_bn=False):
        if newmodel:
            model = self.newmodel
        else:
            model = self.model
        if ckpt:
            model.load_state_dict(ckpt)
        if cal_bn:
            model.train()
            # for idx,(data, target) in enumerate(tqdm(self.trainset, total=len(self.trainset))):
            for idx, (data, target) in enumerate(self.trainset):
                data, target = data.cuda(), target.cuda()
                if idx==100:
                    break
                with torch.no_grad():
                    _=model(data)
            # print("calibrate bn done.")
        model.eval()
        test_loss = 0
        correct = 0
        # for data, target in tqdm(self.testset, total=len(self.testset)):
        for idx,(data, target) in enumerate(self.testset):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.testset.dataset)
        return correct.item() / float(len(self.testset.dataset))

    def train(self):
        self.newmodel.train()
        avg_loss = 0.
        train_acc = 0.
        for batch_idx, (data, target) in tqdm(enumerate(self.trainset), total=len(self.trainset)):
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.newmodel(data)
            loss = F.cross_entropy(output, target)
            avg_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss.backward()
            self.optimizer.step()

    def finetune(self):
        best_prec1 = 0
        for epoch in range(3):
            self.train()
            prec1 = self.test()
            self.scheduler.step(prec1)
            lr_current = self.optimizer.param_groups[0]['lr']
            print("currnt lr:{},current prec:{}".format(lr_current,prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                ckptfile = os.path.join(self.args.savepath, 'ft_model_best.pth.tar')
            else:
                ckptfile = os.path.join(self.args.savepath, 'ft_checkpoint.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.newmodel.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }, ckptfile)
        return best_prec1
    def clone_model(self):
        blockidx = 0
        for name, m0 in self.newmodel.named_modules():
            if not isinstance(m0, InvertedResidual) and not isinstance(m0, conv_bn_relu) and not isinstance(m0, nn.Linear) and not isinstance(m0, conv_bn_relu):
                continue
            block = self.blocks[blockidx]
            curstatedict = block.statedict
            if blockidx == 0:
                inputmask = torch.arange(block.inputchannel)
            assert name == block.layername
            if isinstance(block, CB):
                # conv(1weight)->bn(4weight)->relu
                assert len(curstatedict) == (1 + 4)
                block.clone2module(m0, inputmask)
                inputmask = block.prunemask
            if isinstance(block, InverRes):
                # dw->project or expand->dw->project
                assert len(curstatedict) in (10, 15)
                block.clone2module(m0, inputmask)
                inputmask = torch.arange(block.outputchannel)
            if isinstance(block, FC):
                block.clone2module(m0)
            blockidx += 1
            if blockidx > (len(self.blocks) - 1): break

    def get_flops(self,model):
        from thop import clever_format, profile
        input = torch.randn(1, 3, 32, 32).cuda()
        flops, params = profile(model, inputs=(input,), verbose=False)
        return flops,params