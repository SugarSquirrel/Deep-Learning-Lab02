# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)
        # 解码器部分
        self.decoder4 = self.up_conv_block(1024, 512)
        self.decoder3 = self.up_conv_block(512, 256)
        self.decoder2 = self.up_conv_block(256, 128)
        self.decoder1 = self.up_conv_block(128, 64)
        # 最终输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))
        enc5 = self.encoder5(F.max_pool2d(enc4, kernel_size=2))
        # 解码器
        dec4 = self.decoder4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.conv_block(dec4.size(1), 512)(dec4)
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.conv_block(dec3.size(1), 256)(dec3)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.conv_block(dec2.size(1), 128)(dec2)
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.conv_block(dec1.size(1), 64)(dec1)
        # 最终输出
        return self.final_conv(dec1)
