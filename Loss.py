import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class ENSCrossEntropyLoss(nn.Module):
    def __init__(self, repeat=None, device=None):
        super(ENSCrossEntropyLoss, self).__init__()
        self.smooth = 1e-7
        self.repeat = repeat
        self.device = device

    def forward(self, inputs, label):
        ### make input ###
        inputs = inputs.view(inputs.shape[0], self.repeat, 3, inputs.shape[2], inputs.shape[3])
        inputs = F.softmax(inputs, dim=2)
        x = torch.flatten(inputs, start_dim=3)
        
        ### make label ###
        labels = torch.flatten(label, start_dim=1)
        label_l = labels.long()
        onehot_label = torch.eye(2)[label_l]
        onehot_label = onehot_label.unsqueeze(dim=3)
        onehot_label = torch.repeat_interleave(onehot_label, self.repeat, dim=3)
        onehot_label = onehot_label.permute((0,3,2,1))
        onehot_label = onehot_label.cuda(self.device)

        ### loss ###
        loss = -1*(onehot_label*torch.log(x+self.smooth))

        return torch.mean(loss)

