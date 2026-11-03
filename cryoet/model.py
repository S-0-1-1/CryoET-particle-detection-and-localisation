import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASSES, WIDTHS


def conv_bn_act(cin, cout, stride=1):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm3d(cout), nn.SiLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(cin, cout, stride), nn.Conv3d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm3d(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce = nn.Conv3d(cin, cout, 1, bias=False)
        self.block = ResBlock(cout + cskip, cout)

    def forward(self, x, skip):
        x = self.reduce(F.interpolate(x, scale_factor=2, mode='nearest'))
        return self.block(torch.cat([x, skip], 1))


class HeatmapUNet(nn.Module):
    def __init__(self, n_classes=len(CLASSES), widths=WIDTHS):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = conv_bn_act(1, 16)
        self.enc1 = nn.Sequential(ResBlock(16, w0, 2), ResBlock(w0, w0))
        self.enc2 = nn.Sequential(ResBlock(w0, w1, 2), ResBlock(w1, w1))
        self.enc3 = nn.Sequential(ResBlock(w1, w2, 2), ResBlock(w2, w2))
        self.enc4 = nn.Sequential(ResBlock(w2, w3, 2), ResBlock(w3, w3))
        self.dec3 = UpBlock(w3, w2, w2)
        self.dec2 = UpBlock(w2, w1, w1)
        self.dec1 = UpBlock(w1, w0, w0)
        self.neck = conv_bn_act(w0, w0)
        self.head = nn.Conv3d(w0, n_classes, 1)
        self.aux = nn.Conv3d(w0, 1, 1)
        nn.init.constant_(self.head.bias, -4.6)

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.dec3(e4, e3)
        d = self.dec2(d, e2)
        d = self.neck(self.dec1(d, e1))
        return self.head(d), self.aux(d)
