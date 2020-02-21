import math
import torch.nn as nn


from models.slimmableops import USBatchNorm2d, USConv2d, USLinear, make_divisible
import USconfig as FLAGS

class USInvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(USInvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                USConv2d(
                    inp, expand_inp, 1, 1, 0, bias=False,
                    ratio=[1, expand_ratio]),
                USBatchNorm2d(expand_inp, ratio=expand_ratio),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
                USConv2d(
                    expand_inp, expand_inp, 3, stride, 1, groups=expand_inp,
                    depthwise=True, bias=False,
                    ratio=[expand_ratio, expand_ratio]),
                USBatchNorm2d(expand_inp, ratio=expand_ratio),
                nn.ReLU6(inplace=True),

                USConv2d(
                    expand_inp, outp, 1, 1, 0, bias=False,
                    ratio=[expand_ratio, 1]),
                USBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class USMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, input_size=224):
        super(USMobileNetV2, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.features = []

        width_mult = FLAGS.width_mult_range[-1]
        # head
        assert input_size % 32 == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(
            1280 * width_mult) if width_mult > 1.0 else 1280
        first_stride = 1
        self.features.append(
            nn.Sequential(
                USConv2d(
                    3, channels, 3, first_stride, 1, bias=False,
                    us=[False, True]),
                USBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        USInvertedResidual(channels, outp, s, t))
                else:
                    self.features.append(
                        USInvertedResidual(channels, outp, 1, t))
                channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                USConv2d(
                    channels, self.outp, 1, 1, 0, bias=False,
                    us=[True, False]),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        # avg_pool_size = input_size // 32
        # self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.outp, num_classes))
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
if __name__ == '__main__':
    import torch
    from thop import profile,clever_format
    model=USMobileNetV2()
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    inp=torch.ones(1,3,32,32)
    def count_USConv2d(m, x, y):
        x = x[0]
        kernel_ops = m.weight.size()[2:].numel()  # Kw x Kh
        bias_ops = 1 if m.bias is not None else 0

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

        m.total_ops += torch.Tensor([int(total_ops)])
    flops, params = profile(model, inputs=(inp,),custom_ops={
        USConv2d:count_USConv2d
    }, verbose=False)
    flops,params=clever_format([flops,params])
    print(flops,params)
    out=model(inp)
    print(out.shape)