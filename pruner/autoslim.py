from pruner.BasePruner import BasePruner
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from pruner.Block import *
from models import *
from collections import OrderedDict
import time
import torch.optim as optim
class AutoSlimPruner(BasePruner):
    def __init__(self, model, newmodel, testset, trainset, optimizer, args, prunestep=16,constrain=200e6,savelog='logs'):
        super().__init__(model, newmodel, testset, trainset, optimizer, args)
        self.prunestep = prunestep
        self.constrain = constrain
        if not os.path.isdir(savelog):
            os.mkdir(savelog)
        self.savelog=savelog
    def prune(self,ckpt=None):
        super().prune()
        # gather BN weights
        block_channels = OrderedDict()
        for idx, b in enumerate(self.blocks):
            if b.bnscale is None:
                block_channels.update({idx: None})
            else:
                block_channels.update({
                    idx:
                        {'numch': b.bnscale.shape[0],
                         'flops': 0,
                         'params': 0, }
                })
                b.prunemask = torch.arange(0, b.bnscale.shape[0])
        block_channels.update({
            'cur_acc':0
        })
        if ckpt is not None:
            block_channels=torch.load(ckpt)
            for idx, b in enumerate(self.blocks):
                if block_channels[idx] is None:
                    b.prunemask=None
                else:
                    b.prunemask=torch.arange(0,block_channels[idx]['numch'])
        prune_iter=0
        s=time.time()
        while(1):
            prune_results = []
            for idx, b in enumerate(self.blocks):
                if (block_channels[idx] == None or (block_channels[idx]['numch'] - self.prunestep)<=0):
                    prune_results.append(-1)
                    continue
                b.prunemask = torch.arange(0, block_channels[idx]['numch'] - self.prunestep).cuda()
                assert b.prunemask.shape[0]>0
                self.clone_model()
                flops, params = self.get_flops(self.newmodel)
                block_channels[idx]['flops'] = flops
                block_channels[idx]['params'] = params
                self.newmodel.apply(bn_calibration_init)
                accpruned = self.test(newmodel=True, cal_bn=True)
                print("flops:{}  params:{} acc:{}".format(flops,params,accpruned))
                prune_results.append(accpruned)
                # reset prunemask
                b.prunemask = torch.arange(0, block_channels[idx]['numch']).cuda()
                break
            pick_idx=prune_results.index(max(prune_results))
            if block_channels[pick_idx]['flops']<self.constrain:
                break
            block_channels[pick_idx]['numch']-=self.prunestep
            self.blocks[pick_idx].prunemask=torch.arange(0, block_channels[pick_idx]['numch']).cuda()
            print("iteration {}: prune {},current flops:{},current params:{} ,results:{},spend {}sec".format(
                prune_iter,pick_idx,block_channels[pick_idx]['flops'],block_channels[pick_idx]['params'],max(prune_results),round(time.time()-s)))
            block_channels['cur_acc']=max(prune_results)
            torch.save(block_channels,'{}/{}.pth'.format(self.savelog,prune_iter))
            prune_iter+=1

    def finetune(self,retrain=True,ckpt=None):
        super().prune()
        if ckpt is not None:
            block_channels=torch.load(ckpt)
            for idx, b in enumerate(self.blocks):
                if block_channels[idx] is None:
                    pass
                else:
                    b.prunemask=torch.arange(0,block_channels[idx]['numch'])
            self.clone_model()
        self.newmodel._initialize_weights()
        self.optimizer = optim.SGD(self.newmodel.parameters(), lr=self.args.finetunelr, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=0)
        best_prec1=0.1
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train()
            prec1 = self.test()
            scheduler.step(epoch)
            lr_current = self.optimizer.param_groups[0]['lr']
            print("epoch {} currnt lr:{},acc:{}".format(epoch, lr_current,prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                ckptfile = os.path.join(self.savelog, 'model_best.pth.tar')
            else:
                ckptfile = os.path.join(self.savelog, 'checkpoint.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.newmodel.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }, ckptfile)
        return best_prec1