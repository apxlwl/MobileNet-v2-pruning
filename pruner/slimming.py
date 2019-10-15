from pruner.BasePruner import BasePruner
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from pruner.Block import *
from models import MobileNetV2, InvertedResidual, sepconv_bn, conv_bn


class SlimmingPruner(BasePruner):
    def __init__(self, model, newmodel, testset, trainset, optimizer, args, pruneratio=0.1):
        super().__init__(model, newmodel, testset, trainset, optimizer, args)
        self.pruneratio = pruneratio

    def prune(self):
        blocks = []
        for midx, (name, module) in enumerate(self.model.named_modules()):
            idx = len(blocks)
            if isinstance(module, InvertedResidual):
                blocks.append(InverRes(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
            if isinstance(module, conv_bn):
                blocks.append(CB(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
            if isinstance(module, nn.Linear):
                blocks.append(FC(name, idx, idx - 1, idx + 1, list(module.state_dict().values())))
        #gather BN weights
        bns=[]
        for b in blocks:
            if b.bnscale is not None and b.layername is not 'features.18':
               bns.extend(b.bnscale.tolist())
        bns=torch.Tensor(bns)
        y, i = torch.sort(bns)
        thre_index = int(bns.shape[0]*self.pruneratio)
        thre = y[thre_index]
        thre = thre.cuda()
        pruned_bn=0
        for b in blocks:
            if isinstance(b, CB):
                mask = b.bnscale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername,mask.shape[0] - torch.sum(mask),mask.shape[0]))
            if isinstance(b, InverRes):
                if b.numlayer == 3:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                    print("{}:{}/{} pruned".format(b.layername,mask.shape[0] - torch.sum(mask),mask.shape[0]))

        blockidx = 0

        for name, m0 in self.newmodel.named_modules():
            if not isinstance(m0, InvertedResidual) and not isinstance(m0, conv_bn) and not isinstance(m0, nn.Linear):
                continue
            block = blocks[blockidx]
            curstatedict = block.statedict
            if blockidx == 0:
                inputmask = torch.arange(block.inputchannel)
            assert name == block.layername
            if isinstance(block, CB):
                # conv(1weight)->bn(4weight)->relu
                assert len(curstatedict) == (1 + 4)
                if name == 'features.18':
                    block.clone2module(m0, inputmask, keepoutput=True)
                else:
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
            if blockidx > (len(blocks) - 1): break
        print("Slimming Pruner done")
