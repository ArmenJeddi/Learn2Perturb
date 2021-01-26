'''
    resnet version 1 architecture
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from down_sample import DownsampleA


class L2P_Basicblock_V1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(L2P_Basicblock_V1, self).__init__()
        self.add_noise = False

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

        if planes == 16:
            self.sigma_map = nn.Parameter(torch.ones((16, 32, 32)) * 0.25 , requires_grad=True)
        elif planes == 32:
            self.sigma_map = nn.Parameter(torch.ones((32, 16, 16)) * 0.25, requires_grad=True)
        else:
            self.sigma_map = nn.Parameter(torch.ones((64, 8, 8)) * 0.25, requires_grad=True)

    def put_noise(self, mode = True):
        self.add_noise = mode

    def forward(self, x):
        residual = x
        basicblock = F.relu(self.bn_a(self.conv_a(x)))

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + basicblock

        if self.add_noise:
            self.normal_noise = self.sigma_map.clone().normal_(0,1)
            self.perf = self.normal_noise * self.sigma_map
            self.final_noise = self.perf.expand(out.size())

            out += self.final_noise
        
        return F.relu(out, inplace=True)


class L2P_ResNet_V1(nn.Module):

    def __init__(self, block, depth, num_classes=10):

        super(L2P_ResNet_V1, self).__init__()
        
        assert (depth - 2) % 6 == 0, '(depth - 2) % 6 should equal zero'
        layer_blocks = (depth - 2) // 6
        self.add_noise = False

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.cn1_sigma_map = nn.Parameter(torch.ones((16, 32, 32)) * 0.25 , requires_grad=True)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def put_noise(self, mode = True):
        self.add_noise = mode

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = self.bn_1(x)
        if self.add_noise:
            self.cn1_normal_noise = torch.randn_like(self.cn1_sigma_map)
            self.cn1_perf = (self.cn1_normal_noise * self.cn1_sigma_map).cuda(0)
            self.cn1_final_noise = self.cn1_perf.expand(x.size())
            x += self.cn1_final_noise

        x = F.relu(x, inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# (depth - 2) % 6 should equal 0
def l2p_resnet_v1(depth = 20, num_classes=10): 
    model = L2P_ResNet_V1(L2P_Basicblock_V1, depth=depth, num_classes=num_classes)
    return model
