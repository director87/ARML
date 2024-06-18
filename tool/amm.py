import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out

def ExponentialProjection(x, lamda=1.0):
    # med = torch.median(x).detach()
    # if x[:] > 0:
    x_out = lamda * torch.exp(- lamda* x)
    # else:
    #     x_out = torch.exp(-(x - mean) ** 2 / (2 * mean ** 2 * x)) * math.sqrt(1 / (2 * math.pi * math.pow(-x, 3)))
    return x_out

def PsuedoRayleighProjection(x, std):
    x_out = (1 / std) * torch.exp(- x ** 2 / (2 * std ** 2))
    return x_out

def LaplaceProjection(x, mean, beta=1.0):
    x_out = (1 / (2 * beta)) * torch.exp(- torch.abs(x - mean) / beta)
    return x_out

def conversetanh(x):
    x_out = 1 / (2 * math.pi) * 1 / torch.cosh(-x / 2)**2
    return x_out+1



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        ).cuda()
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # Gauss modulation
        mean = torch.mean(channel_att_sum).detach()
        std = torch.std(channel_att_sum).detach()
        # print(channel_att_sum.shape)
        sig = 1 - F.softmax(torch.max(channel_att_sum) - channel_att_sum, dim=0)
        # print(sig)
        # scale = GaussProjection(channel_att_sum, mean, std).unsqueeze(2).unsqueeze(3).expand_as(x)
        # scale = PsuedoRayleighProjection(channel_att_sum, std).unsqueeze(2).unsqueeze(3).expand_as(x)
        # scale = conversetanh(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = ExponentialProjection(channel_att_sum, lamda=0.8).unsqueeze(2).unsqueeze(3).expand_as(x)
        # scale = LaplaceProjection(channel_att_sum, mean, beta=1.0).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print(scale)

        # scale = scale / torch.max(scale)
        return x * scale
        # return x * channel_att_sum


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.pool = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_pool = self.pool(x)
        x_out = self.spatial(x_pool)

        # Gauss modulation
        mean = torch.mean(x_out).detach()
        std = torch.std(x_out).detach()
        sig = 1 - F.softmax(torch.max(x_out) - x_out, dim=0)
        # scale = GaussProjection(x_out, mean, std)
        scale = ExponentialProjection(x_out, lamda=0.8)
        # scale = LaplaceProjection(x_out, mean, beta=1.0)


        # scale = scale / torch.max(scale)
        return x * scale
        # return x * x_out


class AMM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(AMM, self).__init__()
        self.ChannelAMM = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialAMM = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelAMM(x)
        # print(x_out.shape)
        x_out = self.SpatialAMM(x_out)
        # print(x_out.shape)

        # x_out = self.SpatialAMM(x)
        return x_out

class SCA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(SCA, self).__init__()
        self.ChannelAMM_avg = ChannelGate(gate_channels, reduction_ratio, pool_types=['avg'])
        self.ChannelAMM_max = ChannelGate(gate_channels, reduction_ratio, pool_types=['max'])
        self.SpatialAMM = SpatialGate()

    def forward(self, x):
        x_avg = self.ChannelAMM_avg(x)
        x_max = self.ChannelAMM_max(x)
        x_c = (x_avg + x_max)/2
        # x_c = (x_avg + x_max)
        x_s = self.SpatialAMM(x)
        # x_sc = self.SpatialAMM((self.ChannelAMM_avg(x) + self.ChannelAMM_max(x))/2)
        # x_sc = self.SpatialAMM((self.ChannelAMM_avg(x)))
        x_sc = self.SpatialAMM((self.ChannelAMM_avg(x) + self.ChannelAMM_max(x)))
        x_out = (x_sc + x_s + x_c) / 3
        # x_out = x_s + x_c
        # print(x_out.shape)
        # x_out = self.SpatialAMM(x_out)
        # x_out = self.SpatialAMM(x)
        return x_out