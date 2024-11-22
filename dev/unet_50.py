import torch
from torch import nn

from utils import bilinear_kernel


class ResNetEncoder50(nn.Module):
    def __init__(self, resnet50):
        super(ResNetEncoder50, self).__init__()
        self.resnet = resnet50
        self.encoder_layers = list(self.resnet.children())[:-2]  # no avgpool nor fc
        self.encoder_block0 = nn.Sequential(*self.encoder_layers[:4])  # kind of init
        self.encoder_block1 = self.encoder_layers[4]
        self.encoder_block2 = self.encoder_layers[5]
        self.encoder_block3 = self.encoder_layers[6]
        self.encoder_block4 = self.encoder_layers[7]

    def forward(self, x):
        x0 = self.encoder_block0(x)
        x1 = self.encoder_block1(x0)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)
        return x, x0, x1, x2, x3, x4


class ResNetDecoder50(nn.Module):
    def __init__(self):
        super(ResNetDecoder50, self).__init__()

        def _upconv_block(in_channels, out_channels):
            w = bilinear_kernel(in_channels, out_channels, 3)
            conv_transpose = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
            conv_transpose.weight.data.copy_(w)
            conv_transpose.bias.data.zero_()
            return nn.Sequential(
                conv_transpose,
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # decoder
        self.upconv4 = _upconv_block(2048, 1024)
        self.upconv3 = _upconv_block(2048, 512)
        self.upconv2 = _upconv_block(1024, 256)
        self.upconv1 = _upconv_block(512, 64)
        # x0 must match d1 shape
        self.upconv0 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # final upconv
        self.upconv = _upconv_block(128, 64)

    def forward(self, X):
        _, x0, x1, x2, x3, x4 = X
        d4 = self.upconv4(x4)
        d3 = self.upconv3(torch.cat((d4, x3), dim=1))
        d2 = self.upconv2(torch.cat((d3, x2), dim=1))
        d1 = self.upconv1(torch.cat((d2, x1), dim=1))
        x0_up = self.upconv0(x0)
        d0 = self.upconv(torch.cat((d1, x0_up), dim=1))
        return d0


class ResNetUNet50(nn.Module):
    def __init__(self, resnet50):
        super(ResNetUNet50, self).__init__()
        self.encoder = ResNetEncoder50(resnet50)
        self.decoder = ResNetDecoder50()
        self.final_conv = nn.Sequential(
            nn.Conv2d(67, 32, kernel_size=1), nn.Conv2d(32, 3, kernel_size=1)
        )

    def forward(self, x):
        x_encoder = self.encoder(x)
        x = self.decoder(x_encoder)
        x = self.final_conv(torch.cat((x_encoder[0], x), dim=1))
        return x
