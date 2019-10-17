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
        self.bnscale=None
    def clone2module(self, module: nn.Module, inputmask,keepoutput:bool):
        raise NotImplementedError

    def _cloneBN(self,bn,statedict,mask):
        assert isinstance(bn,nn.BatchNorm2d)
        bn.weight.data = statedict[0][mask.tolist()].clone()
        bn.bias.data = statedict[1][mask.tolist()].clone()
        bn.running_mean = statedict[2][mask.tolist()].clone()
        bn.running_var = statedict[3][mask.tolist()].clone()
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
        self.bnscale=self.statedict[1].abs().clone()
    def clone2module(self, module: nn.Module, inputmask,keepoutput=False):
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)]
        temp = self.statedict[0][:, inputmask.tolist(), :, :]
        if keepoutput:
            modulelayers[0].weight.data = temp.clone()
            self._cloneBN(modulelayers[1],self.statedict[1:5],torch.arange(self.statedict[1].shape[0]))
        else:
            modulelayers[0].weight.data = temp[self.prunemask.tolist(), :, :, :].clone()
            self._cloneBN(modulelayers[1], self.statedict[1:5], self.prunemask)

class InverRes(Baselayer):
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        super().__init__(layername, id, input, output, statedict)
        self.inputchannel = self.statedict[0].shape[1]
        self.outputchannel = self.statedict[-1].shape[0]
        self.numlayer = len(self.statedict) // 5
        if self.numlayer==3:
            self.bnscale=self.statedict[1].abs().clone()
        else:
            self.bnscale=None
    def clone2module(self, module: nn.Module, inputmask,keepoutput=False):
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)]
        if self.numlayer == 2:
            modulelayers[0].weight.data = self.statedict[0][inputmask.tolist(), :, :, :].clone()
            modulelayers[0].groups=inputmask.shape[0]
            self._cloneBN(modulelayers[1],self.statedict[1:5],inputmask)

            modulelayers[2].weight.data = self.statedict[5][:,inputmask.tolist(),:,:].clone()
            self._cloneBN(modulelayers[3], self.statedict[6:10], torch.arange(self.statedict[6].shape[0]))

        if self.numlayer == 3:

            temp = self.statedict[0][:, inputmask.tolist(), :, :]
            modulelayers[0].weight.data = temp[self.prunemask.tolist(), :, :, :].clone()
            self._cloneBN(modulelayers[1],self.statedict[1:5],self.prunemask)

            modulelayers[2].weight.data = self.statedict[5][self.prunemask.tolist(),:,:,:]
            modulelayers[2].groups=self.prunemask.shape[0]
            self._cloneBN(modulelayers[3],self.statedict[6:10],self.prunemask)

            modulelayers[4].weight.data = self.statedict[10][:, self.prunemask.tolist(), :, :]
            self._cloneBN(modulelayers[5], self.statedict[11:15], torch.arange(self.statedict[11].shape[0]))

class FC(Baselayer):
    def __init__(self, layername: str, id: int, input: int, output: int, statedict: list):
        super().__init__(layername, id, input, output, statedict)
        self.inputchannel = self.statedict[0].shape[1]
        self.outputchannel = self.statedict[0].shape[0]

    def clone2module(self, module: nn.Module,inputmask=None,keepoutput=False):
        modulelayers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        modulelayers[0].weight.data=self.statedict[0].clone()
        modulelayers[0].bias.data=self.statedict[1].clone()
