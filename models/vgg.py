import torch
import torch.nn as nn
import math

from collections import OrderedDict
from models.baseblock import conv_bn_relu
defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, depth=19, cfg=None, n_class=10, input_size=224,):
        super(VGG, self).__init__()

        if cfg == None:
            cfg = defaultcfg[depth]

        self.feature = self.make_layers(cfg)
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(cfg[-1], n_class)
        self._initial_weights()

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        block = conv_bn_relu

        for channel in cfg:
            if channel == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(block(inp=in_channels, oup=channel, kernel=3, stride=1, padding=1,relu='relu'))
                in_channels = channel

        return nn.Sequential(*layers)

    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.normal_(0, 1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



if __name__ == '__main__':
    import thop

    vgg = VGG()
    input = torch.randn(1, 3, 32, 32)
    output = vgg(input)
    print(output.shape)

    flops, params = thop.profile(vgg, inputs=(input,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params)
