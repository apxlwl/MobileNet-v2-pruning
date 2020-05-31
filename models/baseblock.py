import torch
import torch.nn as nn
from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.conv = nn.Sequential(OrderedDict([
            ('conv_1', conv3x3(inplanes, planes, stride)),
            ('conv_1_bn', norm_layer(planes)),
            ('conv_1_relu', nn.ReLU(inplace=True)),
            ('conv_2', conv3x3(planes, planes)),
            ('conv_2_bn', norm_layer(planes))
        ]))

    def forward(self, x):
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(x + out)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.conv = nn.Sequential(OrderedDict([
            ('conv_1', conv1x1(inplanes, width)),
            ('conv_1_bn', norm_layer(width)),
            ('conv_1_relu', nn.ReLU(inplace=True)),
            ('conv_2', conv3x3(width, width, stride, groups, dilation)),
            ('conv_2_bn', norm_layer(width)),
            ('conv_2_relu', nn.ReLU(inplace=True)),
            ('conv_3', conv1x1(width, planes * self.expansion)),
            ('conv_3_bn', norm_layer(planes * self.expansion))
        ]))
    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            # s=self.branch_main(x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

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
                ('relu', nn.ReLU(inplace=True))
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
