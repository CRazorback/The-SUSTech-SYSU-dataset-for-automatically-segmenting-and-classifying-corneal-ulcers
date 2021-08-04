import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Model


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resize=None):
        super().__init__()
        self.resize = resize

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if self.resize == 'up':
            self.f = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if self.resize == 'down':
            self.f = nn.MaxPool2d(2)

    def forward(self, x1, x2=None):
        if self.resize: 
            x1 = self.f(x1)
        if self.resize == 'up': 
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x1, x2], axis=1)
        x = self.conv(x1)      
        return x


class UNet(Model):
    def __init__(self, input_channels, multi_scale=False):
        super(UNet, self).__init__()
        self.multi_scale = multi_scale

        self.fe = ConvBlock(input_channels, 32)
        self.d1 = ConvBlock(32, 64, 'down')
        self.d2 = ConvBlock(64, 128, 'down')
        self.d3 = ConvBlock(128, 256, 'down')
        self.d4 = ConvBlock(256, 512, 'down')
        self.u1 = ConvBlock(512 + 256, 256, 'up')
        self.u2 = ConvBlock(256 + 128, 128, 'up')
        self.u3 = ConvBlock(128 + 64, 64, 'up')
        self.u4 = ConvBlock(64 + 32, 32, 'up')
        self.pred = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x, inference=False):
        # multi-scale
        if self.multi_scale and inference:
            return self.mscale_inference(x)

        x0 = self.fe(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.u1(x4, x3)
        x6 = self.u2(x5, x2)
        x7 = self.u3(x6, x1)
        x8 = self.u4(x7, x0)
        # prediction
        logits = self.pred(x8)

        return logits