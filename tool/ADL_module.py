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
        # print(cams)
        # print(cams.shape)
        # dilation_out1 = nn.Conv2d(4, 20, 1, bias=False).cuda()
        # print(dilation_out.weight.shape)
        # cams = F.conv2d(cams, dilation_out1.weight, dilation=2)
        # print(cams.shape)
        # dilation_out2 = nn.Conv2d(20, 4, 1, bias=False).cuda()
        # cams = F.conv2d(cams, dilation_out2.weight, dilation=3)
        # print(cams)
        cams = F.relu(cams)
        # print(cams.shape)
        cams2 = cams
        N, C, H, W = cams.size()
        # print(N, C, H, W)
        cam_mean = torch.mean(cams, dim=1)  # N 28 28
        cam_mean_2 = torch.mean(cams, dim=(2, 3), keepdim=True)
        # print(cam_mean.shape, cam_mean_2.shape)

        zero = torch.zeros_like(cam_mean)
        one = torch.ones_like(cam_mean)
        mean_drop_cam = zero
        for i in range(C):
            sub_cam = cams[:, i, :, :]
            # print(sub_cam.shape)
            sub_cam_max = torch.max(sub_cam.view(N, -1), dim=-1)[0].view(N, 1, 1)
            thr = (sub_cam_max * gama)
            thr = thr.expand(sub_cam.shape)
            sub_cam_with_drop = torch.where(sub_cam > thr, zero, sub_cam)
            mean_drop_cam = mean_drop_cam + sub_cam_with_drop
        mean_drop_cam = mean_drop_cam / 4
        mean_drop_cam = torch.unsqueeze(mean_drop_cam, dim=1)
        # print(mean_drop_cam.shape)


        zero_2 = torch.zeros_like(cam_mean_2)
        one_2 = torch.ones_like(cam_mean_2)
        mean_drop_cam2 = zero_2
        # print(mean_drop_cam2.shape)
        for j in range(H):
            for k in range(W):
                sub_cam2 = cams2[:, :, j, k]
                # print(sub_cam.shape)
                sub_cam_max2 = torch.max(sub_cam2.view(N, -1), dim=-1, keepdim=True)[0]
                # print(sub_cam_max2.shape)
                thr2 = (sub_cam_max2 * gama)
                thr2 = thr2.expand(sub_cam2.shape)
                sub_cam_with_drop2 = torch.where(sub_cam2 > thr2, zero_2, sub_cam2)
                mean_drop_cam2 = mean_drop_cam2 + sub_cam_with_drop2
                # mean_drop_cam2[:, :, j, k] = sub_cam_with_drop2
        # print(mean_drop_cam2.shape)
        mean_drop_cam2 = mean_drop_cam2 / (H * W)
        # print(mean_drop_cam2.shape)
        mean_drop_cam2 = F.avg_pool2d(mean_drop_cam2, mean_drop_cam2.size()[2:])
        # print(mean_drop_cam2.shape)
        linear_layer = nn.Linear(C, 4096).cuda()
        mean_drop_cam2 = linear_layer(mean_drop_cam2.view(N, C))
        mean_drop_cam2 = torch.unsqueeze(mean_drop_cam2, dim=2)
        mean_drop_cam2 = torch.unsqueeze(mean_drop_cam2, dim=3)
        # cams2 = cams2 * mean_drop_cam2
        # print(mean_drop_cam2.shape)
        # print(cams2.shape)
        # print(x.shape)

        # max_pool = torch.max(cams, dim=1, keepdim=True)[0]
        # # print(max_pool.shape)
        # avg_pool = torch.mean(cams, dim=1, keepdim=True)
        # # print(avg_pool.shape)
        # y = torch.cat([max_pool, avg_pool], dim=1)
        # # print(y.shape)
        # conv = nn.Conv2d(2, 1, kernel_size=5, padding=2).cuda()
        # y = conv(y)
        # print(y)
        # print(y, nn.Sigmoid()(y))
        # print(nn.Sigmoid()(y).shape)
        y = self.avgpool(x)
        y = self.Conv_Squeeze(y)
        y = self.relu(y)
        y = self.Conv_Excitation(y)
        y = self.norm(y)
        y = x * y.expand_as(y)
        y = y.cuda()
        # print(x.shape)



        # x = 0.5 * x.cuda() * mean_drop_cam + 0.5 * x.cuda() * mean_drop_cam2
        x = x.cuda() * mean_drop_cam
        # print(x)
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
