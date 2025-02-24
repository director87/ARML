import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool import infer_utils
import network.resnet38d
from tool.ADL_module import Attention_Module
from tool.amm import AMM
from tool.cca import CrissCrossAttention


class Net(network.resnet38d.Net):
    def __init__(self, gama, n_class):
        super().__init__()
        
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.Attention_Module = Attention_Module()
        self.gama = gama
        self.amm = AMM(4096, 4)
        self.cca = CrissCrossAttention(4096)

        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        # self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        # self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        # self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        # torch.nn.init.kaiming_normal_(self.f8_3.weight)
        # torch.nn.init.kaiming_normal_(self.f8_4.weight)
        # torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)


        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]
        # self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8]


    def forward(self, x, enable_PDA, enable_AMM, enable_NAEA, enable_MARS):

        x = super().forward(x)  #[8,4096,28,28]
        gama = self.gama
        if enable_PDA:
            x = self.Attention_Module(x, self.fc8.weight, gama)    #[8,4096,28,28]
        elif enable_AMM:
            x = self.amm(x)
        elif enable_MARS:
            x = self.cca(x)
            x = self.amm(x)
        else:
            x = x

        cams = F.conv2d(x, self.fc8.weight)
        cams = F.relu(cams)
        x = self.dropout7(x)    #[8,4096,28,28]
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)  #[8,4096,1,1]
        feature = x
        feature = feature.view(feature.size(0), -1)#[8,4096]
        x = self.fc8(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        
        return x, feature, y, cams

    def forward_cam(self, x):
        x = super().forward(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Net_CAM(network.resnet38d.Net):
    def __init__(self, n_class):
        super().__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]


    def forward(self, x):
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.pool(x)
        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        return y

    def forward_cam(self, x):
        x_ = super().forward(x)
        x_pool = F.avg_pool2d(x_, kernel_size=(x_.size(2), x_.size(3)), padding=0)
        x = F.conv2d(x_, self.fc8.weight)
        cam = F.relu(x)
        y = self.fc8(x_pool)
        y = y.view(y.size(0), -1)
        y = torch.sigmoid(y)
        
        return cam, y


    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
        #     acc += (logit.argmax(dim=1) == label.view(-1)).sum().float()
        #     num += label.size(0)

        # return loss / batch_size, acc / num
