import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class Network1(nn.Module):
    def __init__(self, inputs=1, output=3, filte=16):
        super(Network1, self).__init__()
        self.conv1 = nn.Conv2d(inputs,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,4,2,padding=1) 
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,64,4,2,padding=1)
        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.conv6 = nn.Conv2d(64,128,4,2,padding=1)
        self.conv7 = nn.Conv2d(128,128,3,padding=1)
        self.deconv1 = nn.ConvTranspose2d(256,64,4,2,padding=1)
        self.conv8 = nn.Conv2d(64,64,3,1,padding=1)
        self.deconv2 = nn.ConvTranspose2d(128,32,4,2,padding=1)
        self.conv9 = nn.Conv2d(32,32,3,1,padding=1)
        self.deconv3 = nn.ConvTranspose2d(96,16,4,2,padding=1)
        self.conv10 = nn.Conv2d(16,filte,3,1,padding=1)
        self.conv_f = nn.Conv2d(filte,output,1,1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(128)
        self.bnde1 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(64) 
        self.bnde2 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(32)
        self.bnde3 = nn.BatchNorm2d(16)
        self.bn10 = nn.BatchNorm2d(16)
        
        self.dp1  = nn.Dropout()
        self.dp2  = nn.Dropout()
        self.dp3  = nn.Dropout()
        self.dp4  = nn.Dropout()
        self.dp5  = nn.Dropout()
        self.dp6  = nn.Dropout()
        self.dp7  = nn.Dropout()
        self.dp8  = nn.Dropout()
        self.dp9  = nn.Dropout()
        self.dp10  = nn.Dropout()
        self.dp11  = nn.Dropout()
        self.dp12  = nn.Dropout()
        

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h = self.dp1(h1)
        h2 = F.relu(self.bn2(self.conv2(h)))
        h = self.dp2(h2)
        h3 = F.relu(self.bn3(self.conv3(h)))
        h = self.dp3(h3)
        h4 = F.relu(self.bn4(self.conv4(h)))
        h = self.dp4(h4)
        h5 = F.relu(self.bn5(self.conv5(h)))
        h = self.dp5(h5)
        h6 = F.relu(self.bn6(self.conv6(h)))
        h = self.dp6(h6)
        h7 = F.relu(self.bn7(self.conv7(h)))
        h = self.dp7(h7)
        h = torch.cat((h,h6),1)
        h = F.relu(self.bnde1(self.deconv1(h)))
        h = self.dp8(h)
        h = F.relu(self.bn8(self.conv8(h)))
        h = self.dp9(h)
        h = torch.cat((h,h4),1)
        h = F.relu(self.bnde2(self.deconv2(h)))
        h = self.dp10(h)
        h = F.relu(self.bn9(self.conv9(h)))
        h = self.dp11(h)
        h = torch.cat((h,h2),1)
        h = F.relu(self.bnde3(self.deconv3(h)))
        h = self.dp12(h)
        hp = F.relu(self.bn10(self.conv10(h)))
        h = self.conv_f(hp)

        return h, hp


class Network2(nn.Module):
    def __init__(self, inputs=1, output=3, filte=16):
        super(Network2, self).__init__()
        self.conv1 = nn.Conv2d(inputs,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,4,2,padding=1) 
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,64,4,2,padding=1)
        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.conv6 = nn.Conv2d(64,128,4,2,padding=1)
        self.conv7 = nn.Conv2d(128,128,3,padding=1)
        self.deconv1 = nn.ConvTranspose2d(256,64,4,2,padding=1)
        self.conv8 = nn.Conv2d(64,64,3,1,padding=1)
        self.deconv2 = nn.ConvTranspose2d(128,32,4,2,padding=1)
        self.conv9 = nn.Conv2d(32,32,3,1,padding=1)
        self.deconv3 = nn.ConvTranspose2d(96,16,4,2,padding=1)
        self.conv10 = nn.Conv2d(16,16,3,1,1)
        self.conv_f = nn.Conv2d(16,output,1,1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(128)
        self.bnde1 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(64) 
        self.bnde2 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(32)
        self.bnde3 = nn.BatchNorm2d(16)
        self.bn10 = nn.BatchNorm2d(16)
        
        self.dp1  = nn.Dropout()
        self.dp2  = nn.Dropout()
        self.dp3  = nn.Dropout()
        self.dp4  = nn.Dropout()
        self.dp5  = nn.Dropout()
        self.dp6  = nn.Dropout()
        self.dp7  = nn.Dropout()
        self.dp8  = nn.Dropout()
        self.dp9  = nn.Dropout()
        self.dp10  = nn.Dropout()
        self.dp11  = nn.Dropout()
        self.dp12  = nn.Dropout()


    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h = self.dp1(h1)
        h2 = F.relu(self.bn2(self.conv2(h)))
        h = self.dp2(h2)
        h3 = F.relu(self.bn3(self.conv3(h)))
        h = self.dp3(h3)
        h4 = F.relu(self.bn4(self.conv4(h)))
        h = self.dp4(h4)
        h5 = F.relu(self.bn5(self.conv5(h)))
        h = self.dp5(h5)
        h6 = F.relu(self.bn6(self.conv6(h)))
        h = self.dp6(h6)
        h7 = F.relu(self.bn7(self.conv7(h)))
        h = self.dp7(h7)
        h = torch.cat((h,h6),1)
        h = F.relu(self.bnde1(self.deconv1(h)))
        h = self.dp8(h)
        h = F.relu(self.bn8(self.conv8(h)))
        h = self.dp9(h)
        h = torch.cat((h,h4),1)
        h = F.relu(self.bnde2(self.deconv2(h)))
        h = self.dp10(h)
        h = F.relu(self.bn9(self.conv9(h)))
        h = self.dp11(h)
        h = torch.cat((h,h2),1)
        h = F.relu(self.bnde3(self.deconv3(h)))
        h = self.dp12(h)
        h = F.relu(self.bn10(self.conv10(h)))
        h = self.conv_f(h)

        return h
        
        
class EnsembleNet(nn.Module):
    def __init__(self, inputs=1):
        super(EnsembleNet, self).__init__()
        self.conv1 = nn.Conv3d(inputs,1,1,1)

    def forward(self, x):
        h = self.conv1(x)
        return h.sum(dim=1)





