#coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

import os
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from Net import Network1, Network2, EnsembleNet
from Loss import ENSCrossEntropyLoss
from Mydataset import RPECellDataloader
import utils as ut


### IoU ###
def IoU(output, target, label):
    output = np.array(output)
    target = np.array(target)
    seg = np.argmax(output,axis=1)
    seg = seg.flatten()
    target = target.flatten() 
    mat = confusion_matrix(target, seg, labels=label)
    iou_den = (mat.sum(axis=1) + mat.sum(axis=0) - np.diag(mat))
    iou = np.array(np.diag(mat) ,dtype=np.float32) / np.array(iou_den, dtype=np.float32)

    return iou


### training ###
def train(epoch):
    sum_loss = 0
    
    model1.train()
    model2.train()
    Ensuble.train()
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        targets = targets.long()

        ### first step ###
        output1, hp = model1(inputs)
        loss1 = criterion(output1, targets)
        loss1 = loss1.mean(dim=[1,2], keepdim=True)
        output1 = output1.unsqueeze(1)
        
        output1n = inputs + hp
        output1n = output1n.unsqueeze(2)
        
        ### second step ###
        for i in range(args.filter):
            output2 = model2(torch.sigmoid(output1n[:,i]))
            loss2 = criterion(output2, targets)
            loss2 = loss2.mean(dim=[1,2], keepdim=True)
            loss1 = torch.cat([loss1, loss2], dim=1)
            output2 = output2.unsqueeze(1)
            output1 = torch.cat([output1, output2], dim=1)

        ### ensemble ###
        output3 = Ensuble(output1)
        loss3 = criterion(output3, targets)
        loss3 = loss3.mean()
        
        ### all loss ###
        loss = torch.sum(loss1, dim=[1,2])
        loss = loss.mean()
        loss = loss + loss3


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        sum_loss += loss.item()
 
    return sum_loss/(batch_idx+1)



### validation ###
def validation(epoch):
    sum_loss = 0
    model1.eval()
    model2.eval()
    Ensuble.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, leave=False)):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()
                
            ### first step ###
            output1, hp = model1(inputs)
            loss1 = criterion(output1, targets)
            loss1 = loss1.mean(dim=[1,2], keepdim=True)
            output1 = output1.unsqueeze(1)
                        
            output1n = inputs + hp
            output1n = output1n.unsqueeze(2)
                        
            ### second step ###
            for i in range(args.filter):
                output2 = model2(torch.sigmoid(output1n[:,i]))
                loss2 = criterion(output2, targets)
                loss2 = loss2.mean(dim=[1,2], keepdim=True)
                loss1 = torch.cat([loss1, loss2], dim=1)
                output2 = output2.unsqueeze(1)
                output1 = torch.cat([output1, output2], dim=1)

            ### ensemble ###
            output3 = Ensuble(output1)
            loss3 = criterion(output3, targets)
            loss3 = loss3.mean()
            
            ### all loss ###
            loss = torch.sum(loss1, dim=[1,2])
            loss = loss.mean()
            loss = loss + loss3

            sum_loss += loss.item() 

            ### save output ###
            output = F.softmax(output3, dim=1)
            inputs = inputs.cpu().numpy()
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()
            
            for i in range(args.batchsize):
                predict.append(output[i])
                answer.append(targets[i])


        ### IoU ###
        iou = IoU(predict, answer, label=[0,1])


    return sum_loss/(batch_idx+1), iou.mean()


##### main #####
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datapath', default='./Dataset')
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=str, default=-1)
    parser.add_argument('--cross', type=int, default=0)
    parser.add_argument('--filter', type=int, default=1)
    args = parser.parse_args()
    gpu_flag = args.gpu

    ### device ###
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    ### save ###
    if not os.path.exists("{}".format(args.out)):
      	os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "model")):
      	os.mkdir(os.path.join("{}".format(args.out), "model"))
      	
    PATH_1 = "{}/trainloss.txt".format(args.out)
    PATH_2 = "{}/testloss.txt".format(args.out)
    PATH_3 = "{}/IoU.txt".format(args.out)
    
    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass


    ### seed ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



    ### data loader ###
    data_train = RPECellDataloader(root = args.datapath, 
                                   cross = args.cross, 
                                   dataset_type='train',
                                   transform=ut.ExtCompose([ut.ExtToTensor()]))
                                   
    data_val = RPECellDataloader(root = args.datapath, 
                                 cross = args.cross,     
                                 dataset_type='val',
                                 transform=ut.ExtCompose([ut.ExtToTensor()]))
                                   
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=4)



    # networks #
    model1 = Network1(inputs=1, output=2).cuda(device)
    model2 = Network2(inputs=1, output=2).cuda(device)
    Ensuble = EnsembleNet(inputs=args.filter+1).cuda(device)
    
    
    # optimizer #
    params = list(model1.parameters()) + list(model2.parameters()) + list(Ensuble.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    
    # loss function #
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_ens = ENSCrossEntropyLoss(repeat=args.filter)


    ### training & validation ###
    sample = 0
    sample_loss = 10000000
    
    for epoch in range(args.num_epochs):
        loss_train = train(epoch)
        loss_test, mm = validation(epoch)

        print("epoch %d / %d" % (epoch+1, args.num_epochs))
        print('train loss: %.4f' % loss_train)
        print('test loss : %.4f' % loss_test)
        print(" Mean IoU : %.4f" % mm)
        print("")

        ### save ###
        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, mm))


        ### model save ###
        PATH1 ="{}/model/model1_train.pth".format(args.out)
        torch.save(model1.state_dict(), PATH1)
        PATH2 ="{}/model/model2_train.pth".format(args.out)
        torch.save(model2.state_dict(), PATH2)
        PATH3 ="{}/model/ENS_train.pth".format(args.out)
        torch.save(Ensuble.state_dict(), PATH3)

        ### best miou ###
        if mm >= sample:
           sample = mm
           PATH1_best ="{}/model/model1_bestiou.pth".format(args.out)
           torch.save(model1.state_dict(), PATH1_best)
           PATH2_best ="{}/model/model2_bestiou.pth".format(args.out)
           torch.save(model2.state_dict(), PATH2_best)
           PATH3_best ="{}/model/ENS_bestiou.pth".format(args.out)
           torch.save(Ensuble.state_dict(), PATH3_best)

        ### best test loss ###
        if loss_test < sample_loss:
           sample_loss = loss_test
           PATH1_best ="{}/model/model1_bestloss.pth".format(args.out)
           torch.save(model1.state_dict(), PATH1_best)
           PATH2_best ="{}/model/model2_bestloss.pth".format(args.out)
           torch.save(model2.state_dict(), PATH2_best)
           PATH3_best ="{}/model/ENS_bestloss.pth".format(args.out)
           torch.save(Ensuble.state_dict(), PATH3_best)

