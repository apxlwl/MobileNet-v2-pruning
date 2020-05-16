import torch
import torch.nn as nn
from collections import OrderedDict
class conv_bn_relu(nn.Module):
    def __init__(self,inp, oup, kernel,stride,padding,relu='relu6'):
        super(conv_bn_relu, self).__init__()
        if relu=='relu6':
            self.convbn=nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(inp, oup, kernel, stride, padding, bias=False)),
                ('bn', nn.BatchNorm2d(oup)),
                ('relu', nn.ReLU6(inplace=True))
            ]))
        else:
            self.convbn = nn.Sequential(OrderedDict([
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
