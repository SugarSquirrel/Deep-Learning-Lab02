# Implement your ResNet34_UNet model here
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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.relu(out)
        return out

class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet34Encoder, self).__init__()
        self.in_channels = 64 

        # ResNet 第一層conv
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet34 的四層 BasicBlock
        self.layer1 = self._make_layer(BasicBlock, out_channels=64, num_blocks=3, stride=1)  # 64x56x56
        self.layer2 = self._make_layer(BasicBlock, out_channels=128, num_blocks=4, stride=2) # 128x28x28
        self.layer3 = self._make_layer(BasicBlock, out_channels=256, num_blocks=6, stride=2) # 256x14x14
        self.layer4 = self._make_layer(BasicBlock, out_channels=512, num_blocks=3, stride=2) # 512x7x7

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []

        # encoder
        x = self.conv1(x)  # 64x112x112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64x56x56
        
        skip_connections.append(x)  # Skip 1
        x = self.layer1(x)  # 64x56x56

        skip_connections.append(x)  # Skip 2
        x = self.layer2(x)  # 128x28x28

        skip_connections.append(x)  # Skip 3
        x = self.layer3(x)  # 256x14x14

        skip_connections.append(x)  # Skip 4
        x = self.layer4(x)  # 512x7x7
        
        # for i in range(len(skip_connections)):
        #     print("> skip_connections[{}]: {}".format(i, skip_connections[i].shape)) # 最上到最下 64, 128, 256, 512

        return x, skip_connections[::-1]  # 倒序return skip connections

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34_UNet, self).__init__()
        self.downs = ResNet34Encoder(in_channels=in_channels)  # ResNet34 當 Encoder
        
        # Up
        # for idx, feature in enumerate(reversed(features)):
        #     # self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
        #     # self.ups.append(DoubleConv(feature*2, feature))
        #     # if idx == 0 or idx == 1:
        #     #     self.ups.append(nn.ConvTranspose2d(feature, feature, kernel_size=2, stride=2))
        #     #     self.ups.append(DoubleConv(feature, feature))
        #     # else:
        #     self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
        #     self.ups.append(DoubleConv(feature*2, feature))
        
        # up: 512x7x7 -> 256x14x14, conv: 512 -> 256
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        # up: 256x14x14 -> 128x28x28, conv: 256 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        # up: 128x28x28 -> 64x56x56, conv: 128 -> 64
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        # up: 64x56x56 -> 64x112x112, conv: 128 -> 64
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # Final輸出層, out = 1

    def forward(self, x):
        skip_connections = []

        x, skip_connections = self.downs(x)

        x = self.up1(x)
        skip_connection = skip_connections[0] # 256
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        concar_skip = torch.cat((skip_connection, x), dim=1)
        x = self.conv1(concar_skip)

        x = self.up2(x)
        skip_connection = skip_connections[1] # 128
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        concar_skip = torch.cat((skip_connection, x), dim=1)
        x = self.conv2(concar_skip)

        x = self.up3(x)
        skip_connection = skip_connections[2] # 64
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        concar_skip = torch.cat((skip_connection, x), dim=1)
        x = self.conv3(concar_skip)

        x = self.up4(x)
        skip_connection = skip_connections[3] # 64
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        concar_skip = torch.cat((skip_connection, x), dim=1)
        x = self.conv4(concar_skip)

        '''
        for i in range(0, len(self.ups), 2):
            #skip_connections
            #1024, 512, 2, 2
            #1024, 512
            #512, 256, 2, 2
            #512, 256
            #256, 128, 2, 2
            #256, 128
            #128, 64, 2, 2
            #128, 64
            print("ups[i]: ", i)
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2] # 0, 0, 1, 1, 2, 2, 3, 3
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concar_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concar_skip)
        '''
        return self.final_conv(x)
    
if __name__ == "__main__":
    model = ResNet34_UNet(in_channels=3, out_channels=1)
    x = torch.randn((32, 3, 384, 384))  # Batch size = 32, RGB image, 256x256
    preds = model(x)
    print(preds.shape)  # 要是是 (32, 1, 256, 256)