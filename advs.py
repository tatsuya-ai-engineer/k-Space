import torchattacks

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models import *
from utils import progress_bar




def imshow(img, adv_name, save_path):
    # 非正規化する
    img = img / 2 + 0.5
    # torch.Tensor型からnumpy.ndarray型に変換する
    # print(type(img)) # <class 'torch.Tensor'>
    npimg = img.to('cpu').detach().numpy().copy()
    # print(type(npimg))    
    # 形状を（RGB、縦、横）から（縦、横、RGB）に変換する
    # print(npimg.shape)
    npimg = np.transpose(npimg, (1, 2, 0))
    # print(npimg.shape)
    # 画像を表示する
    plt.imshow(npimg)
    plt.savefig(save_path)
    # plt.show()
    
    plt.clf()
    plt.close()




device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = ["resnet50", "google"]
atk_names = ["pgd", "fgsm", "deepfool", 'onepix']


### load original image ######################
max_idx = [98, 85, 1, 70, 73, 47, 89, 65, 31, 47]

load_path = '/home/ueda-tatsuya/デスクトップ/k-Spectrum/pytorch-cifar_forU/output_dicts/origined_CIFAR10_datas_v2.pkl'
with open(load_path, mode='rb') as f:
    dataset = pickle.load(f)

classes = dataset.keys()

imgs = np.empty((0, 3, 32, 32))
lbs = np.empty((0))
for max_num, c in enumerate(classes):
    tensor_img = dataset[c][max_idx[max_num]].unsqueeze(dim=0) 
    img = tensor_img.to('cpu').detach().numpy().copy()
    
    imgs = np.append(imgs, img, axis=0)
    lbs = np.append(lbs, max_num)

imgs = torch.from_numpy(imgs.astype(np.float32)).clone()
lbs = torch.from_numpy(lbs.astype(np.int64)).clone()



for model_name in models:
    ### load model ######
    if model_name == "vgg19":
        net = VGG('VGG19')
    elif model_name == "resnet50":
        net = ResNet50()
    elif model_name == "google":
        net = GoogLeNet()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    checkpoint = torch.load(f'/home/ueda-tatsuya/デスクトップ/k-Spectrum/pytorch-cifar_forU/checkpoint/{model_name}_ckpt.pth')
    net.load_state_dict(checkpoint['net'])



    for adv_name in tqdm(atk_names):
        
        for i in range(1,11,1):
            
            if adv_name == "pgd":
                atk = torchattacks.PGD(net, eps=i/255, alpha=2/255, steps=40)
            elif adv_name == "fgsm":
                atk = torchattacks.FGSM(net, eps=i/1000)
            elif adv_name == "deepfool":
                atk = torchattacks.DeepFool(net, steps=50, overshoot=i/100)
            elif adv_name == "onepix":
                atk = torchattacks.OnePixel(net, pixels=i, steps=75, popsize=400, inf_batch=128)
            
            adv_images = atk(imgs, lbs)
            imshow(torchvision.utils.make_grid(adv_images), adv_name+str(i),
                save_path=f"./adv_imgs/{model_name}/{adv_name+str(i)}.png")
            
            
            ### test ###
            if model_name == "vgg19":
                test_net = VGG('VGG19')
            elif model_name == "resnet50":
                test_net = ResNet50()
            elif model_name == "google":
                test_net = GoogLeNet()

            test_net = test_net.to(device)
            if device == 'cuda':
                test_net = torch.nn.DataParallel(test_net)
                cudnn.benchmark = True
            
            for aug_n in range(5):
                checkpoint = torch.load(f'/home/ueda-tatsuya/デスクトップ/k-Spectrum/pytorch-cifar_forU/checkpoint/{model_name}_aug{aug_n}_ckpt.pth')
                test_net.load_state_dict(checkpoint['net'])
                
                soft = test_net(adv_images)
                pred = torch.max(soft.data, 1)
                
                out_dict = {
                    "softmax": soft,
                    "prediction list": pred[1]
                }
                
                save_path = f'./adv_imgs/{model_name}/{adv_name+str(i)}_aug{aug_n}.pkl'
                with open(save_path, 'wb') as p:
                    pickle.dump(out_dict , p)

