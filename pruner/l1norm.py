from pruner.BasePruner import BasePruner
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from pruner.Block import *
from models import MobileNetV2, InvertedResidual, sepconv_bn, conv_bn


class l1normPruner(BasePruner):
    def __init__(self, model, newmodel, testset, trainset, optimizer, args, pruneratio=0.1):
        super().__init__(model, newmodel, testset, trainset, optimizer, args)
        self.pruneratio = pruneratio

    def prune(self):
        super().prune()
        for b in self.blocks:
            if isinstance(b, CB):
                pruneweight = torch.sum(torch.abs(b.statedict[0]), dim=(1, 2, 3))
                numkeep = int(pruneweight.shape[0] * (1 - self.pruneratio))
                _ascend = torch.argsort(pruneweight)
                _descend = torch.flip(_ascend, (0,))[:numkeep]
                mask = torch.zeros_like(pruneweight).long()
                mask[_descend] = 1
                b.prunemask = torch.where(mask == 1)[0]
            if isinstance(b, InverRes):
                if b.numlayer == 3:
                    pruneweight = torch.sum(torch.abs(b.statedict[0]), dim=(1, 2, 3))
                    numkeep = int(pruneweight.shape[0] * (1 - self.pruneratio))
                    _ascend = torch.argsort(pruneweight)
                    _descend = torch.flip(_ascend, (0,))[:numkeep]
                    mask = torch.zeros_like(pruneweight).long()
                    mask[_descend] = 1
                    b.prunemask = torch.where(mask == 1)[0]
        self.clone_model()
        print("l1 norm Pruner done")
