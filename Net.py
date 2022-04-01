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
        
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)


    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h = F.dropout(h1)
        h2 = F.relu(self.bn2(self.conv2(h)))
        h = F.dropout(h2)
        h3 = F.relu(self.bn3(self.conv3(h)))
        h = F.dropout(h3)
        h4 = F.relu(self.bn4(self.conv4(h)))
        h = F.dropout(h4)
        h5 = F.relu(self.bn5(self.conv5(h)))
        h = F.dropout(h5)
        h6 = F.relu(self.bn6(self.conv6(h)))
        h = F.dropout(h6)
        h7 = F.relu(self.bn7(self.conv7(h)))
        h = F.dropout(h7)
        h = torch.cat((h,h6),1)
        h = F.relu(self.bnde1(self.deconv1(h)))
        h = F.dropout(h)
        h = F.relu(self.bn8(self.conv8(h)))
        h = F.dropout(h)
        h = torch.cat((h,h4),1)
        h = F.relu(self.bnde2(self.deconv2(h)))
        h = F.dropout(h)
        h = F.relu(self.bn9(self.conv9(h)))
        h = F.dropout(h)
        h = torch.cat((h,h2),1)
        h = F.relu(self.bnde3(self.deconv3(h)))
        h = F.dropout(h)
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


    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h = F.dropout(h1)
        h2 = F.relu(self.bn2(self.conv2(h)))
        h = F.dropout(h2)
        h3 = F.relu(self.bn3(self.conv3(h)))
        h = F.dropout(h3)
        h4 = F.relu(self.bn4(self.conv4(h)))
        h = F.dropout(h4)
        h5 = F.relu(self.bn5(self.conv5(h)))
        h = F.dropout(h5)
        h6 = F.relu(self.bn6(self.conv6(h)))
        h = F.dropout(h6)
        h7 = F.relu(self.bn7(self.conv7(h)))
        h = F.dropout(h7)
        h = torch.cat((h,h6),1)
        h = F.relu(self.bnde1(self.deconv1(h)))
        h = F.dropout(h)
        h = F.relu(self.bn8(self.conv8(h)))
        h = F.dropout(h)
        h = torch.cat((h,h4),1)
        h = F.relu(self.bnde2(self.deconv2(h)))
        h = F.dropout(h)
        h = F.relu(self.bn9(self.conv9(h)))
        h = F.dropout(h)
        h = torch.cat((h,h2),1)
        h = F.relu(self.bnde3(self.deconv3(h)))
        h = F.dropout(h)
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





