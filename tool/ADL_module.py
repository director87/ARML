import torch
from torch import nn
import torch.nn.functional as F


class Attention_Module(nn.Module):
    '''
    input: torch.tensor: c*4096*28*28
    ouput:4*1 feature vector
    '''

    def __init__(self):
        super(Attention_Module, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # b c 1 1
        self.relu = nn.ReLU()
        self.Conv_Squeeze = nn.Conv2d(4096, 2048, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
        self.Conv_Excitation = nn.Conv2d(2048, 4096, kernel_size=1, bias=False)


    def forward(self, x, fc_weights, gama):
        cams = F.conv2d(x, fc_weights, dilation=1).cuda()
        cams = F.relu(cams)
        N, C, H, W = cams.size()
        cam_mean = torch.mean(cams, dim=1)  # N 28 28

        zero = torch.zeros_like(cam_mean)
        one = torch.ones_like(cam_mean)
        mean_drop_cam = zero
        for i in range(C):
            sub_cam = cams[:, i, :, :]
            sub_cam_max = torch.max(sub_cam.view(N, -1), dim=-1)[0].view(N, 1, 1)
            thr = (sub_cam_max * gama)
            thr = thr.expand(sub_cam.shape)
            sub_cam_with_drop = torch.where(sub_cam > thr, zero, sub_cam)
            mean_drop_cam = mean_drop_cam + sub_cam_with_drop
        mean_drop_cam = mean_drop_cam / 4
        mean_drop_cam = torch.unsqueeze(mean_drop_cam, dim=1)

        x = x.cuda() * mean_drop_cam
        return x


if __name__ == '__main__':
    x = torch.rand(8, 4096, 28, 28)
    fc8 = nn.Conv2d(4096, 4, 1, bias=False)
    torch.nn.init.xavier_uniform_(fc8.weight)
    # print(fc8.weight)
    AM = Attention_Module()
    x = AM(x, fc8.weight, 1)
    print(x.shape)
    dropout7 = torch.nn.Dropout2d(0.5)
    x = dropout7(x)
    print(x.shape)
    x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
    print(x.shape)
