'''
    resnet version 1 architecture
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2P_Basicblock_V2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(L2P_Basicblock_V2, self).__init__()
        self.add_noise = False

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            
        if planes == 64:
            self.sigma_map = nn.Parameter(torch.ones((64, 32, 32))*0.25, requires_grad=True)
        elif planes == 128:
            self.sigma_map = nn.Parameter(torch.ones((128, 16, 16))*0.25, requires_grad=True)
        elif planes == 256:
            self.sigma_map = nn.Parameter(torch.ones((256, 8, 8))*0.25, requires_grad=True)
        else:
            self.sigma_map = nn.Parameter(torch.ones((512, 4, 4))*0.25, requires_grad=True)

    def put_noise(self, mode = True):
        self.add_noise = mode
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        short = self.shortcut(x)

        out += short

        if self.add_noise:
            self.normal_noise = self.sigma_map.clone().normal_(0,1)
            self.perf = self.normal_noise * self.sigma_map
            self.final_noise = self.perf.expand(out.size())

            out += self.final_noise

        out = F.relu(out)

        return out

class L2P_ResNet_V2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(L2P_ResNet_V2, self).__init__()
        self.in_planes = 64

        self.add_noise = False

        self.cn1_sigma_map = nn.Parameter(torch.ones((64, 32, 32)) * 0.25 , requires_grad=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def put_noise(self, mode = True):
        self.add_noise = mode

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.add_noise:
            self.cn1_normal_noise = torch.randn_like(self.cn1_sigma_map)
            self.cn1_perf = (self.cn1_normal_noise * self.cn1_sigma_map).cuda(0)
            self.cn1_final_noise = self.cn1_perf.expand(out.size())
            out += self.cn1_final_noise
        
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def l2p_resnet_v2(num_blocks= [2,2,2,2], num_classes= 10):
    return L2P_ResNet_V2(L2P_Basicblock_V2, num_blocks= num_blocks, num_classes= num_classes)
