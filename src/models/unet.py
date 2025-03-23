# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None, kernel_size=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down  
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # self.in_channels = in_channels
        # self.out_channels = out_channels

        # self.encoder1 = DoubleConv(in_channels, 64)
        # self.encoder2 = DoubleConv(64, 128)
        # self.encoder3 = DoubleConv(128, 256)
        # self.encoder4 = DoubleConv(256, 512)
        # self.encoder5 = DoubleConv(512, 1024)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # self.decoder4 = DoubleConv(1024, 512)
        # self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.decoder3 = DoubleConv(512, 256)
        # self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.decoder2 = DoubleConv(256, 128)
        # self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.decoder1 = DoubleConv(128, 64)

        # self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concar_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concar_skip)
        
        return self.final_conv(x)
        # enc1 = self.encoder1(x)
        # enc2 = self.encoder2(self.pool(enc1))
        # enc3 = self.encoder3(self.pool(enc2))
        # enc4 = self.encoder4(self.pool(enc3))
        # enc5 = self.encoder5(self.pool(enc4))

        # dec4 = self.upconv4(enc5)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        # dec4 = self.decoder4(dec4)

        # dec3 = self.upconv3(dec4)
        # dec3 = torch.cat((dec3, enc3), dim=1)
        # dec3 = self.decoder3(dec3)

        # dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc2), dim=1)
        # dec2 = self.decoder2(dec2)

        # dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.decoder1(dec1)

        # return self.final_conv(dec1)

# def test():
#     x = torch.randn((3, 1, 161, 161))
#     model = UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape == (3, 1, 161, 161)

# if __name__ == "__main__":
#     test()