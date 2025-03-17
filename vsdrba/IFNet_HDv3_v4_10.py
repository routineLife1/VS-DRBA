import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
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
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        if flow is not None:
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        if scale != 1.0:
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear")
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self, scale=1, ensemble=False, b0only=False):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 16, c=192)
        self.b0only = b0only
        if not b0only:
            self.block1 = IFBlock(8 + 4 + 16, c=128)
            self.block2 = IFBlock(8 + 4 + 16, c=96)
            self.block3 = IFBlock(8 + 4 + 16, c=64)
            self.encode = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(32, 8, 4, 2, 1),
            )
        else:
            self.block1 = None
            self.block2 = None
            self.block3 = None
        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble

    def forward(self, img0, img1, timestep, f0, f1):
        img0o = img0.clamp(0.0, 1.0)
        img1o = img1.clamp(0.0, 1.0)
        f0o = f0.clone()
        f1o = f1.clone()
        to = timestep.clone()

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        merged = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):

            scale_current = 1 / self.scale_list[i]
            upscale = self.scale_list[i] / self.scale_list[i + 1] if i < len(block) - 1 else self.scale_list[i]
            scale_next = 1 / self.scale_list[i + 1] if i < len(block) - 1 else 1

            img0, img1, f0, f1, timestep = [F.interpolate(x, scale_factor=scale_current, mode="bilinear") for x in
                                            [img0o, img1o, f0o, f1o, to]]
            if self.b0only:
                merged, _ = block[0](torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=1.0)
                break
            if flow is None:
                flow, mask = block[i](torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=upscale)
                if self.ensemble:
                    f_, m_ = block[i](torch.cat((img1, img0, f1, f0, 1 - timestep), 1), None, scale=upscale)
                    flow = (flow + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (mask + (-m_)) / 2
            else:
                tenFlow_div, backwarp_tenGrid = self.get_grid(f0)
                wf0 = warp(f0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
                wf1 = warp(f1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
                fd, m0 = block[i](
                    torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask), 1), flow, scale=upscale)
                if self.ensemble:
                    f_, m_ = block[i](
                        torch.cat((warped_img1, warped_img0, wf1, wf0, 1 - timestep, -mask), 1),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=upscale,
                    )
                    fd = (fd + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (m0 + (-m_)) / 2
                else:
                    mask = m0
                flow = F.interpolate(flow, scale_factor=upscale, mode="bilinear") * upscale
                flow = flow + fd

            img0, img1 = [F.interpolate(x, scale_factor=scale_next, mode="bilinear") for x in [img0o, img1o]]
            tenFlow_div, backwarp_tenGrid = self.get_grid(img0)
            warped_img0 = warp(img0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
            warped_img1 = warp(img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)

        if not self.b0only:
            mask = torch.sigmoid(mask)
            merged = warped_img0 * mask + warped_img1 * (1 - mask)

        return merged

    def get_grid(self, x):
        _, _, ph, pw = x.shape
        tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=x.device)

        tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=x.device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
        tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=x.device)
        tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
        backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

        return tenFlow_div, backwarp_tenGrid