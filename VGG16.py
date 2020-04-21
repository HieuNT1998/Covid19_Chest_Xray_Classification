import torch
import torch.nn as nn
import torch.nn.functional as F 


class conv_layer_3(nn.Module):          ## 3 conv layer
    def __init__(self,channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in,channel_out,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(channel_out,channel_out,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(channel_out,channel_out,kernel_size=3,stride=1,padding=1)
    
    def forward(self,xb):
        h = F.relu(self.conv1(xb))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h

class conv_layer_2(nn.Module):         ## 2 conv layer
    def __init__(self,channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in,channel_out,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(channel_out,channel_out,kernel_size=3,stride=1,padding=1)
    
    def forward(self,xb):
        h = F.relu(self.conv1(xb))
        h = F.relu(self.conv2(h))
        return h

class conv_layer_1(nn.Module):         ## 1 conv layer
    def __init__(self,channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in,channel_out,kernel_size=3,stride=1,padding=1)
    
    def forward(self,xb):
        h = F.relu(self.conv1(xb))
        return h

class Flatten(nn.Module):             
    def forward(self, input):
        return input.view(input.size(0), -1)

class VGG_16(nn.Module): 
    def __init__(self):
        super().__init__()
        self.batnom0 = nn.BatchNorm2d(3)               ## 224 * 224 * 3
        
        self.conv_layer1 = conv_layer_1(3,64)          ## 224 * 224 * 64
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv_layer2 = conv_layer_2(64,128)        ## 112 * 112 * 128
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv_layer3 = conv_layer_3(128,256)       ## 56 * 56 * 256
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv_layer4 = conv_layer_3(256,512)       ## 28 * 28 * 512
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv_layer5 = conv_layer_3(512,512)       ## 14 * 14 * 512
        self.pool5 = nn.MaxPool2d(8)

        self.flatten = Flatten()                       ## flatten 7 * 7 * 512 => 25088 * 1 

        self.dense1 = nn.Linear(25088,4096)            ## 25088 -> 4096
        self.dense2 = nn.Linear(4096,4096)             ## 4096 -> 4096
        self.dense3 = nn.Linear(4096,2)                ## 4096 -> 2 (class)
        
    def forward(self,xb):
        xb = xb.view(xb.shape[0],xb.shape[3],xb.shape[1],xb.shape[2])
        xb = self.batnom0(xb)
        
        xb = self.conv_layer1(xb)
        xb = self.pool1(xb)

        xb = self.conv_layer2(xb)
        xb = self.pool2(xb)
        
        xb = self.conv_layer3(xb)
        xb = self.pool3(xb)
        
        xb = self.conv_layer4(xb)
        xb = self.pool4(xb)

        xb = self.conv_layer5(xb)
        xb = self.pool5(xb)

        xb = self.flatten(xb)
        
        xb = F.relu(self.dense1(xb))
        xb = F.relu(self.dense2(xb))
        xb = F.softmax(self.dense3(xb),dim=0)

        return xb

def get_model():
    return VGG_16()