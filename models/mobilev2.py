import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from collections import OrderedDict
import math
class conv_bn(nn.Module):
    def __init__(self,inp, oup, kernel,stride,padding):
        super(conv_bn, self).__init__()
        self.convbn=nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inp, oup, kernel, stride, padding, bias=False)),
            ('bn', nn.BatchNorm2d(oup)),
            ('relu', nn.ReLU6(inplace=True))
        ]))
    def forward(self, input):
        return self.convbn(input)

def sepconv_bn(inp,oup,kernel,stride,padding):
    return nn.Sequential(OrderedDict([
        ('sepconv',nn.Conv2d(inp, inp, kernel, stride, padding,groups=inp, bias=False)),
        ('sepbn',nn.BatchNorm2d(inp)),
        ('seprelu',nn.ReLU6(inplace=True)),
        ('pointconv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        ('pointbn', nn.BatchNorm2d(oup)),
        ('pointrelu', nn.ReLU6(inplace=True)),
    ]))

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv=nn.Sequential(OrderedDict([
                ('dw_conv', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                ('dw_bn', nn.BatchNorm2d(hidden_dim)),
                ('dw_relu', nn.ReLU6(inplace=True)),
                ('project_conv', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                ('project_bn', nn.BatchNorm2d(oup))
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict(
                [
                    ('expand_conv',nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                    ('expand_bn', nn.BatchNorm2d(hidden_dim)),
                    ('expand_relu', nn.ReLU6(inplace=True)),
                    ('dw_conv',nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                    ('dw_bn',nn.BatchNorm2d(hidden_dim)),
                    ('dw_relu',nn.ReLU6(inplace=True)),
                    ('project_conv',nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                    ('project_bn',nn.BatchNorm2d(oup))
                ]
            )
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 n_class=10,
                 input_size=224,
                 width_mult=1.,
                 ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        # 1280
        # self.zero_init_residual = zero_init_residual
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.features = [conv_bn(3, input_channel, 3,1,1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_bn(input_channel,self.last_channel, 1,1,0))
        self.features = nn.Sequential(*self.features)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net=MobileNetV2(input_size=32)
    print(net)
    assert 0
    inp=torch.ones(1,3,32,32)
    out=net(inp)
    print(out.shape)