from models.basic_modules import *


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super().__init__()
        # self.W_g = nn.Sequential(
        #     nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(F_int)
        # )
        self.W_g = ConvNorm(F_g, F_int, kernel_size=1, stride=1, activation=False, **kwargs)

        self.W_x = ConvNorm(F_l, F_int, kernel_size=1, stride=2, activation=False, **kwargs)

        self.psi = nn.Sequential(
            ConvNorm(F_int, 1, kernel_size=1, stride=1, activation=False, **kwargs),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * self.upsample(psi)


class ParallelDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2
        self.conv3_0 = ConvNorm(in_channels[0], self.midchannels, 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], self.midchannels, 1, 1, **kwargs)
        self.conv5_0 = ConvNorm(in_channels[2], self.midchannels, 1, 1, **kwargs)

        self.conv4_5 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)
        self.conv3_4 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)

        self.conv_out = nn.Conv3d(3 * self.midchannels, out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        size = x3.shape[2:]

        # first interpolate three feature maps to the same resolution
        f3 = self.conv3_0(x3)  # (None, midchannels, h3, w3)
        f4 = self.conv4_0(F.interpolate(x4, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)
        level5 = self.conv5_0(F.interpolate(x5, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)

        # fuse feature maps
        level4 = self.conv4_5(torch.cat([f4, level5], dim=1))  # (None, midchannels, h3, w3)
        level3 = self.conv3_4(torch.cat([f3, level4], dim=1))  # (None, midchannels, h3, w3)

        fused_out_reduced = torch.cat([level3, level4, level5], dim=1)

        out = self.conv_out(fused_out_reduced)

        return out


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2

        self.conv5_4 = ConvNorm(in_channels[2], in_channels[1], 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], in_channels[1], 3, 1, **kwargs)
        self.conv4_3 = ConvNorm(in_channels[1], in_channels[0], 1, 1, **kwargs)
        self.conv3_0 = ConvNorm(in_channels[0], in_channels[0], 3, 1, **kwargs)

        self.conv_out = nn.Conv3d(in_channels[0], out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        x5_up = self.conv5_4(F.interpolate(x5, size=x4.shape[2:], mode='trilinear', align_corners=False))
        x4_refine = self.conv4_0(x5_up + x4)
        x4_up = self.conv4_3(F.interpolate(x4_refine, size=x3.shape[2:], mode='trilinear', align_corners=False))
        x3_refine = self.conv3_0(x4_up + x3)

        out = self.conv_out(x3_refine)

        return out
