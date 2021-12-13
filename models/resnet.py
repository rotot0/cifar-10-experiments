import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
        if self.stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.stride != 1:
             x = self.bn3(self.downsample(x))
        return F.relu(out + x)

class MyResNet34(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(MyResNet34, self).__init__()
        hid_dim = 64
        self.conv1 = nn.Conv2d(in_channels, hid_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(hid_dim)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer1 = [Block(hid_dim, hid_dim) for _ in range(3)]
        layer2 = [Block(hid_dim, 2*hid_dim, stride=2)] + [Block(2*hid_dim, 2*hid_dim) for _ in range(3)]
        layer3 = [Block(2*hid_dim, 4*hid_dim, stride=2)] + [Block(4*hid_dim, 4*hid_dim) for _ in range(5)]
        layer4 = [Block(4*hid_dim, 8*hid_dim, stride=2)] + [Block(8*hid_dim, 8*hid_dim) for _ in range(2)]
        layers = layer1 + layer2 + layer3 + layer4
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8*hid_dim, num_classes)

        self.init_weights_()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        # x = self.maxpool(x)
    
        x = self.layers(x)

        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def init_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)