import torch
import torch.nn as nn


class Baselayer:
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        self.layername = layername
        self.inputlayer = input
        self.outputlayer = output
        self.layerid = id
        self.inputchannel = 0
        self.outputchannel = 0
        # filter relu
        self.statedict = [s for s in statedict if len(s.shape) != 0]
        self.prunemask = None
        self.bnscale = None
        self.keepoutput = False

    def clone2module(self, module: nn.Module, inputmask, keepoutput: bool):
        raise NotImplementedError

    def _cloneBN(self, bn_module, statedict, mask):
        assert isinstance(bn_module, nn.BatchNorm2d)
        bn_module.weight.data = statedict[0][mask.tolist()].clone()
        bn_module.bias.data = statedict[1][mask.tolist()].clone()
        bn_module.running_mean = statedict[2][mask.tolist()].clone()
        bn_module.running_var = statedict[3][mask.tolist()].clone()

    def _cloneConv(self, conv_module, state_dict, inputmask, prunemask):
        if inputmask is not None:
            temp = state_dict[:, inputmask.tolist(), :, :]
        else:
            temp = state_dict
        if prunemask is not None:
            conv_module.weight.data = temp[prunemask.tolist(), :, :, :].clone()
        else:
            conv_module.weight.data = temp.clone()


    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "name={}, ".format(self.layername)
        s += "id={}, ".format(self.layerid)
        s += "input={}, ".format(self.inputlayer)
        s += "output={},".format(self.outputlayer)
        s += "numweights={},".format(len(self.statedict))
        s += "inchannel={},".format(self.inputchannel)
        s += "outchannel={})".format(self.outputchannel)
        return s


class CB(Baselayer):
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        super().__init__(layername, id, input, output, statedict)
        # 'conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'
        self.inputchannel = self.statedict[0].shape[1]
        self.outputchannel = self.statedict[-1].shape[0]
        self.bnscale = self.statedict[1].abs().clone()

    def clone2module(self, module: nn.Module, inputmask):
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)]
        temp = self.statedict[0][:, inputmask.tolist(), :, :]
        if self.keepoutput:
            modulelayers[0].weight.data = temp.clone()
            self._cloneBN(modulelayers[1], self.statedict[1:5], torch.arange(self.statedict[1].shape[0]))
        else:
            modulelayers[0].weight.data = temp[self.prunemask.tolist(), :, :, :].clone()
            self._cloneBN(modulelayers[1], self.statedict[1:5], self.prunemask)


class InverRes(Baselayer):
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        super().__init__(layername, id, input, output, statedict)
        self.inputchannel = self.statedict[0].shape[1]
        self.outputchannel = self.statedict[-1].shape[0]
        self.numlayer = len(self.statedict) // 5
        if self.numlayer == 3:
            self.bnscale = self.statedict[1].abs().clone()
        else:
            self.bnscale = None

    def clone2module(self, module: nn.Module, inputmask, keepoutput=False):
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)]
        if self.numlayer == 2:
            modulelayers[0].weight.data = self.statedict[0][inputmask.tolist(), :, :, :].clone()
            modulelayers[0].groups = inputmask.shape[0]
            self._cloneBN(modulelayers[1], self.statedict[1:5], inputmask)

            modulelayers[2].weight.data = self.statedict[5][:, inputmask.tolist(), :, :].clone()
            self._cloneBN(modulelayers[3], self.statedict[6:10], torch.arange(self.statedict[6].shape[0]))

        if self.numlayer == 3:
            temp = self.statedict[0][:, inputmask.tolist(), :, :]
            modulelayers[0].weight.data = temp[self.prunemask.tolist(), :, :, :].clone()
            self._cloneBN(modulelayers[1], self.statedict[1:5], self.prunemask)

            modulelayers[2].weight.data = self.statedict[5][self.prunemask.tolist(), :, :, :]
            modulelayers[2].groups = self.prunemask.shape[0]
            self._cloneBN(modulelayers[3], self.statedict[6:10], self.prunemask)

            modulelayers[4].weight.data = self.statedict[10][:, self.prunemask.tolist(), :, :]
            self._cloneBN(modulelayers[5], self.statedict[11:15], torch.arange(self.statedict[11].shape[0]))


class ResBottle(Baselayer):
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        super().__init__(layername, id, input, output, statedict)
        self.inputchannel = self.statedict[0].shape[1]
        self.outputchannel = self.statedict[-1].shape[0]
        self.numlayer = len(self.statedict) // 5
        self.prunemask1 = None
        if self.layername == 'layer1.0':
            for k in self.statedict:
                print(k.shape)
        if self.numlayer == 3:
            self.bnscale = [
                self.statedict[1].abs().clone(),
                self.statedict[6].abs().clone()
            ]
        elif self.numlayer == 4:
            self.bnscale = [
                self.statedict[6].abs().clone(),
                self.statedict[11].abs().clone()
            ]
        else:
            assert NotImplementedError

    def clone2module(self, module: nn.Module, inputmask, keepoutput=False):
        self.prunemask2=self.prunemask
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)]
        # numlayer==3:
        # conv_expand 1*1+ conv 3*3 + conv_proj
        # numlayer==4:
        # conv_downsample +  conv_expand 1*1+ conv 3*3 + conv_proj
        if self.numlayer == 3:
            self._cloneConv(modulelayers[0], self.statedict[0], inputmask=inputmask, prunemask=self.prunemask1)
            self._cloneBN(modulelayers[1], self.statedict[1:5], self.prunemask1)

            self._cloneConv(modulelayers[2],self.statedict[5],self.prunemask1,self.prunemask2)
            self._cloneBN(modulelayers[3], self.statedict[6:10], self.prunemask2)

            self._cloneConv(modulelayers[4],self.statedict[10],self.prunemask2,None)
            self._cloneBN(modulelayers[5], self.statedict[11:15], torch.arange(self.statedict[11].shape[0]))

        if self.numlayer == 4:
            # donot prune downsample
            self._cloneConv(modulelayers[0], self.statedict[0], inputmask=inputmask, prunemask=None)
            self._cloneBN(modulelayers[1], self.statedict[1:5], torch.arange(self.statedict[1].shape[0]))

            self._cloneConv(modulelayers[2],self.statedict[5],inputmask,self.prunemask1)
            self._cloneBN(modulelayers[3], self.statedict[6:10], self.prunemask1)

            self._cloneConv(modulelayers[4],self.statedict[10],self.prunemask1,self.prunemask2)
            self._cloneBN(modulelayers[5], self.statedict[11:15], self.prunemask2)

            self._cloneConv(modulelayers[6],self.statedict[15],self.prunemask2,None)
            self._cloneBN(modulelayers[7], self.statedict[16:20], torch.arange(self.statedict[16].shape[0]))


class ShuffleLayer(Baselayer):
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        super().__init__(layername, id, input, output, statedict)
        self.inputchannel = self.statedict[0].shape[1]
        self.outputchannel = self.statedict[-1].shape[0]
        self.numlayer = len(self.statedict) // 5
        if self.numlayer == 3:
            # self.bnscale = self.statedict[1].abs().clone()
            self.bnscale = self.statedict[6].abs().clone()
        elif self.numlayer == 5:
            # self.bnscale = None
            self.bnscale = self.statedict[1].abs().clone()
        else:
            assert NotImplementedError
        # print(self.numlayer)

    def clone2module(self, module: nn.Module, inputmask, keepoutput=False):
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)]
        if self.numlayer == 5:
            modulelayers[0].weight.data = self.statedict[0][inputmask.tolist(), :, :, :].clone()
            self._cloneBN(modulelayers[1], self.statedict[1:5], inputmask)

            modulelayers[2].weight.data = self.statedict[5][self.prunemask.tolist(), :, :, :].clone()
            modulelayers[2].groups = self.prunemask.shape[0]
            self._cloneBN(modulelayers[3], self.statedict[6:10], torch.arange(self.statedict[6].shape[0]))

            modulelayers[4].weight.data = self.statedict[10][:, self.prunemask.tolist(), :, :].clone()
            self._cloneBN(modulelayers[5], self.statedict[11:15], torch.arange(self.statedict[11].shape[0]))

            # branch_proj: no prunning currently, just copy
            modulelayers[6].weight.data = self.statedict[15].clone()
            modulelayers[6].groups = self.statedict[15].shape[0]
            self._cloneBN(modulelayers[7], self.statedict[16:20], torch.arange(self.statedict[16].shape[0]))

            modulelayers[8].weight.data = self.statedict[20].clone()
            self._cloneBN(modulelayers[9], self.statedict[21:25], torch.arange(self.statedict[21].shape[0]))

        if self.numlayer == 3:
            temp = self.statedict[0][:, inputmask.tolist(), :, :]
            modulelayers[0].weight.data = temp[self.prunemask.tolist(), :, :, :].clone()
            self._cloneBN(modulelayers[1], self.statedict[1:5], self.prunemask)

            modulelayers[2].weight.data = self.statedict[5][self.prunemask.tolist(), :, :, :]
            modulelayers[2].groups = self.prunemask.shape[0]
            self._cloneBN(modulelayers[3], self.statedict[6:10], self.prunemask)

            modulelayers[4].weight.data = self.statedict[10][:, self.prunemask.tolist(), :, :]
            self._cloneBN(modulelayers[5], self.statedict[11:15], torch.arange(self.statedict[11].shape[0]))


class FC(Baselayer):
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        super().__init__(layername, id, input, output, statedict)
        self.inputchannel = self.statedict[0].shape[1]
        self.outputchannel = self.statedict[0].shape[0]

    def clone2module(self, module: nn.Module, inputmask=None, keepoutput=False):
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        modulelayers[0].weight.data = self.statedict[0][:, inputmask.tolist()].clone()
        if len(self.statedict) > 0:
            modulelayers[0].bias.data = self.statedict[1].clone()
