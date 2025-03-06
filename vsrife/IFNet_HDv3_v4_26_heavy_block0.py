import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1\
)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4*13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        # tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear")
        # flow = tmp[:, :4] * scale
        # mask = tmp[:, 4:5]
        # feat = tmp[:, 5:]
        # return flow, mask, feat
        flow = tmp[:, :4]
        return flow

class IFNet(nn.Module):
    def __init__(self, scale=1, ensemble=False):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+32, c=192)
        self.scale_list = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
        if ensemble:
            raise ValueError("rife: ensemble is not supported in v4.26.large")

    def forward(self, img0, img1, timestep, f0, f1):
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        
        return self.block0(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=self.scale_list[0])
