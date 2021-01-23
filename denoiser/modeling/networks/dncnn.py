import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Conv, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Conv_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=True):
        super(Conv_ReLU, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Conv_BN, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Conv_BN_ReLU, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Denoise_Block_BN(nn.Module):
    def __init__(self, input_chnl, output_chnl=None,inner_chnl=64, padding=1, num_of_layers=15, groups=1):
        super(Denoise_Block_BN, self).__init__()
        kernel_size = 3
        num_chnl = inner_chnl
        if output_chnl is None:
            output_chnl = input_chnl
        self.conv_input = nn.Sequential(Conv_BN_ReLU(in_channels=input_chnl, out_channels=num_chnl, kernel_size=kernel_size, padding=padding, groups=groups))
        self.conv_layers = self._make_layers(Conv_BN_ReLU,num_chnl=num_chnl, kernel_size=kernel_size, padding=padding, num_of_layers=num_of_layers-2, groups=groups)
        self.conv_out = nn.Sequential(Conv_BN_ReLU(in_channels=num_chnl, out_channels=output_chnl, kernel_size=kernel_size, padding=padding, groups=groups))

    def _make_layers(self, block, num_chnl, kernel_size, padding, num_of_layers, groups=1):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=num_chnl, out_channels=num_chnl, kernel_size=kernel_size, padding=padding, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_out(self.conv_layers(self.conv_input(x)))


class DnCNN(nn.Module):
    def __init__(self, input_chnl, residual, groups=1):
        super(DnCNN, self).__init__()
        self.residual = residual
        kernel_size = 3
        num_chnl = 64
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_chnl, out_channels=num_chnl,
                              kernel_size=kernel_size, stride=1, padding=1,
                              groups=1, bias=True),
                              nn.ReLU(inplace=True))
        self.dn_block = self._make_layers(Conv_BN_ReLU, kernel_size, num_chnl, num_of_layers=15, bias=False)
        # self.output = nn.Sequential(nn.Conv2d(in_channels=num_chnl, out_channels=input_chnl,
        #                                       kernel_size=kernel_size, stride=1, padding=1,
        #                                       groups=groups, bias=True),
        #                             nn.BatchNorm2d(input_chnl))
        self.output = nn.Conv2d(in_channels=num_chnl, out_channels=input_chnl,
                                              kernel_size=kernel_size, stride=1, padding=1,
                                              groups=groups, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block,  kernel_size, num_chnl, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=num_chnl, out_channels=num_chnl, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.dn_block(x1)
        if self.residual:
            return self.output(x2) + residual
        else:
            return self.output(x2)