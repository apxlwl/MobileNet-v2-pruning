from pruner.BasePruner import BasePruner
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from pruner.Block import *


def css_thresholding(x, OT_DISCARD_PERCENT):
    MIN_SCALING_FACTOR = 1e-18
    x[x < MIN_SCALING_FACTOR] = MIN_SCALING_FACTOR
    x_sorted, _ = torch.sort(x)
    x2 = x_sorted ** 2
    Z = x2.sum()
    energy_loss = 0
    for i in range(x2.shape[0]):
        energy_loss += x2[i]
        if energy_loss / Z > OT_DISCARD_PERCENT:
            break
    th = (x_sorted[i - 1] + x_sorted[i]) / 2 if i > 0 else 0
    return th


class SlimmingPruner(BasePruner):
    def __init__(self, model, newmodel, testset, trainset, optimizer, args, pruneratio=0.1):
        super().__init__(model, newmodel, testset, trainset, optimizer, args)
        self.pruneratio = pruneratio

    def prune(self):
        super().prune()
        bns = []
        thres_perlayer = {}
        for b in self.blocks:
            if b.bnscale is not None:
                # bns.extend(b.bnscale.tolist())
                if isinstance(b.bnscale, list):
                    thres_perlayer[b] = [css_thresholding(scale, OT_DISCARD_PERCENT=1e-2) for scale in b.bnscale]
                else:
                    thres_perlayer[b] = css_thresholding(b.bnscale, OT_DISCARD_PERCENT=1e-2)
        # bns=torch.Tensor(bns)
        # y, i = torch.sort(bns)
        # thre_index = int(bns.shape[0]*self.pruneratio)
        # thre = y[thre_index]
        # thre = thre.cuda()
        pruned_bn = 0
        for b in self.blocks:
            if b.bnscale is None:
                continue
            thre = thres_perlayer[b]
            if isinstance(b, CB):
                mask = b.bnscale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, InverRes):
                if b.numlayer == 3:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                    print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, ResBottle):
                assert len(thre) == 2
                mask = b.bnscale[0].gt(thre[0])
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask1 = torch.where(mask == 1)[0]

                mask = b.bnscale[1].gt(thre[1])
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, ShuffleLayer):
                if b.numlayer == 3:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                    print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
                elif b.numlayer == 5:
                    b.prunemask = torch.arange(b.bnscale.shape[0])
                    print("{}:{}/{} pruned".format(b.layername, 0, b.prunemask.shape[0]))


        if isinstance(self.blocks[-1], FC):  # If the last layer is FC
            # FC layer cannot prune output dimension
            pass

        self.clone_model()
        print("Slimming Pruner done")
