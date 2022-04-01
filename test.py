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
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from Net import Network1, Network2, EnsembleNet
from Loss import ENSCrossEntropyLoss
from Mydataset import RPECellDataloader
import utils as ut


### Evaluation metrics ###
def EvaluationMetrics(output, target, label):
    output = np.array(output)
    target = np.array(target)
    seg = np.argmax(output,axis=2)
    seg = seg.flatten()
    target = target.flatten()
    mat = confusion_matrix(target, seg, labels=label)
    sns.heatmap(mat, annot=True, fmt='.0f', cmap='jet')
    plt.savefig("{}/CM.png".format(args.out))
    mat = np.array(mat).astype(np.float32)
    
    ### IoU ###
    iou_den = (mat.sum(axis=1) + mat.sum(axis=0) - np.diag(mat))
    iou = np.array(np.diag(mat) ,dtype=np.float32) / np.array(iou_den, dtype=np.float32)
    
    ### DSC ###
    mat_all = mat.sum()
    diag_all = np.sum(np.diag(mat))
    fp_all = mat.sum(axis=1)
    fn_all = mat.sum(axis=0)
    tp_tn = np.diag(mat)
    precision = np.zeros((2)).astype(np.float32)
    recall = np.zeros((2)).astype(np.float32)
    dsc = np.zeros((2)).astype(np.float32)

    for i in range(2):
        if (fp_all[i] != 0)and(fn_all[i] != 0):  
            precision[i] = float(tp_tn[i]) / float(fp_all[i])
            recall[i] = float(tp_tn[i]) / float(fn_all[i])
            if (precision[i] != 0)and(recall[i] != 0):  
                dsc[i] = (2.0*precision[i]*recall[i]) / (precision[i]+recall[i])
            else:       
                dsc[i] = 0.0
        else:
            precision[i] = 0.0
            recall[i] = 0.0

    return iou, dsc


def Save_images(img, imgn, fil, out1, out, tar, filte, path, num):
    #print(out1.shape)
    out1 = np.argmax(out1, axis=2)
    out = np.argmax(out, axis=1)
    img = img[0,0]
    imgn = imgn[0]
    fil = fil[0]
    out = out[0]
    tar = tar[0]
    dst1 = np.zeros((img.shape[0],img.shape[1],3))
    dst2 = np.zeros((img.shape[0],img.shape[1],3))

    dst1[out==0] = [0.0,0.0,0.0]
    dst1[out==1] = [1.0,0.0,0.0]
    dst1[out==2] = [0.0,1.0,0.0]

    dst2[tar==0] = [0.0,0.0,0.0]
    dst2[tar==1] = [1.0,0.0,0.0]
    dst2[tar==2] = [0.0,1.0,0.0]

    # save #
    plt.imsave(path + "input/img_{}.png".format(num), img, cmap='gray')
    plt.imsave(path + "seg/seg_{}.png".format(num), dst1)
    plt.imsave(path + "ano/ano_{}.png".format(num), dst2)

    c = 0
    for i in range(filte):
        if not os.path.exists(os.path.join("{}".format(args.out), "image", "seg_{}".format(i+1))):
      	    os.mkdir(os.path.join("{}".format(args.out), "image", "seg_{}".format(i+1)))
        if not os.path.exists(os.path.join("{}".format(args.out), "image", "inputs_{}".format(i+1))):
      	    os.mkdir(os.path.join("{}".format(args.out), "image", "inputs_{}".format(i+1)))
        if not os.path.exists(os.path.join("{}".format(args.out), "image", "filter_{}".format(i+1))):
      	    os.mkdir(os.path.join("{}".format(args.out), "image", "filter_{}".format(i+1)))
      	  
        dst = np.zeros((img.shape[0],img.shape[1],3))
        sss = out1[0,i]
        fff = fil[i]
        dst[sss==0] = [0.0,0.0,0.0]
        dst[sss==1] = [1.0,0.0,0.0]
        dst[sss==2] = [0.0,1.0,0.0]
        iii = imgn[i,0]
        
        # save #
        plt.imsave(path + "seg_{}/seg_{}.png".format(i+1, num), dst)
        plt.imsave(path + "inputs_{}/img_{}.png".format(i+1, num), iii, cmap='gray')
        plt.imsave(path + "filter_{}/f_{}.png".format(i+1, num), fff)
        
    plt.close()


### test ###
def test():
    model_path = "{}/model/model1_bestiou.pth".format(args.out)
    model1.load_state_dict(torch.load(model_path))
    model_path = "{}/model/model2_bestiou.pth".format(args.out)
    model2.load_state_dict(torch.load(model_path))
    model_path = "{}/model/ENS_bestiou.pth".format(args.out)
    Ensuble.load_state_dict(torch.load(model_path))
    model1.eval()
    model2.eval()
    Ensuble.eval()
    
    predict = []
    predict1 = []
    predict2 = []
    answer = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()
                
            ### first step ### 
            output1, hp = model1(inputs)
            output1 = output1.unsqueeze(1)
            
            output1n = inputs + hp
            output1n = output1n.unsqueeze(2)
            
            ### second step ###
            for i in range(args.filter):
                output2 = model2(torch.sigmoid(output1n[:,i]))
                output2 = output2.unsqueeze(1)
                output1 = torch.cat([output1, output2], dim=1)    

            ### ensemble ###
            output3 = Ensuble(output1)
            
            ### save output ###
            output = F.softmax(output3, dim=1)
            
            inputs = inputs.cpu().numpy()
            output1n = output1n.cpu().numpy()
            output = output.cpu().numpy()
            output1 = output1.cpu().numpy()
            hp = hp.cpu().numpy()
            targets = targets.cpu().numpy()
            
            predict.append(output)
            answer.append(targets)

            ### save images ###
            Save_images(inputs, output1n, hp, output1, output, targets, args.filter, "{}/image/".format(args.out), batch_idx+1)


        iou, dsc = EvaluationMetrics(predict, answer, label=[0,1])

        print("Mean IoU = %f ; background = %f ; membrane = %f" % (iou.mean(), iou[0], iou[1]))
        print("Mean DSC = %f ; background = %f ; membrane = %f" % (dsc.mean(), dsc[0], dsc[1]))

        with open(PATH, mode = 'a') as f:
            f.write("%f\t%f\t%f\n" % (iou.mean(), iou[0], iou[1]))
        with open(PATH, mode = 'a') as f:
            f.write("%f\t%f\t%f\n" % (dsc.mean(), dsc[0], dsc[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SR')
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
    if not os.path.exists(os.path.join("{}".format(args.out), "image")):
      	os.mkdir(os.path.join("{}".format(args.out), "image"))
    if not os.path.exists(os.path.join("{}".format(args.out),"image","seg")):
      	os.mkdir(os.path.join("{}".format(args.out),"image", "seg"))
    if not os.path.exists(os.path.join("{}".format(args.out),"image", "ano")):
      	os.mkdir(os.path.join("{}".format(args.out), "image", "ano"))
    if not os.path.exists(os.path.join("{}".format(args.out),"image","input")):
      	os.mkdir(os.path.join("{}".format(args.out), "image", "input"))
      	
    PATH = "{}/predict.txt".format(args.out)
    
    with open(PATH, mode = 'w') as f:
        pass


    ### seed ###
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    ### data loader ###      
    data_test = RPECellDataloader(root = args.datapath, 
                                 cross = args.cross,     
                                 dataset_type='test',
                                 transform=ut.ExtCompose([ut.ExtToTensor()]))
                                   
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=True, num_workers=4)


    ### networks ###
    model1 = Network1(inputs=1, output=2).cuda(device)
    model2 = Network2(inputs=1, output=2).cuda(device)
    Ensuble = EnsembleNet(inputs=args.filter+1).cuda(device)
    

    ### test ###
    test()



