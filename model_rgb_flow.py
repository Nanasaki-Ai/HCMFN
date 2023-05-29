import torch
import torch.nn as nn
from resnet_flow import resnet18 as ResNet


class Model(nn.Module):
    def __init__(self, num_class=120):
        super().__init__()
        self.resnet = ResNet(pretrained=False)
        # self.resnet = ResNet(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                  bias=False)
        self.resnet.fc = nn.Linear(512, num_class) # reznet18


    def forward(self, x_rgb):
        rgb_weighted = x_rgb
        x = self.resnet.conv1(rgb_weighted)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.resnet.fc(x)

        return out

