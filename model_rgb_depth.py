import torch
import torch.nn as nn
from resnet_flow import resnet18 as ResNet


class Model(nn.Module):
    def __init__(self, num_class=60):
        super().__init__()
        # self.resnet = ResNet(pretrained=True)
        self.resnet = ResNet(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.resnet.fc = nn.Linear(512, num_class) # reznet18


    def forward(self, x_rgb):#TODO:2,2,320,320
        rgb_weighted = x_rgb#TODO:2,2,320,320
        x = self.resnet.conv1(rgb_weighted)#TODO:2,64,160,160
        x = self.resnet.bn1(x)#TODO:2,64,160,160
        x = self.resnet.relu(x)#TODO:2,64,160,160
        x = self.resnet.maxpool(x)#TODO:2,64,80,80
        x = self.resnet.layer1(x)#TODO:2,64,80,80
        x = self.resnet.layer2(x)#TODO:2,128,40,40
        x = self.resnet.layer3(x)#TODO:2,256,20,20
        x = self.resnet.layer4(x)#TODO:2,512,10,10
        x = self.resnet.avgpool(x)#TODO:2,512,1,1
        x = torch.flatten(x, 1)#TODO:2,512
        out = self.resnet.fc(x)#TODO:2,60

        return out

