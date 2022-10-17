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




# def imshow(img, adv_name, save_path):
#     # 非正規化する
#     img = img / 2 + 0.5
#     # torch.Tensor型からnumpy.ndarray型に変換する
#     # print(type(img)) # <class 'torch.Tensor'>
#     npimg = img.to('cpu').detach().numpy().copy()
#     # print(type(npimg))    
#     # 形状を（RGB、縦、横）から（縦、横、RGB）に変換する
#     # print(npimg.shape)
#     npimg = np.transpose(npimg, (1, 2, 0))
#     # print(npimg.shape)
#     # 画像を表示する
#     plt.imshow(npimg)
#     plt.savefig(save_path)
#     # plt.show()
    
#     plt.clf()
#     plt.close()



target_dataset = ['IN', 'SIN']
model_name = ['resnet-IN', 'resnet-SIN', 'vit-IN']

classes = ['airplane', 'bear', 'bicycle', 'bird', 
            'boat', 'bottle', 'car', 'cat', 
            'chair', 'clock', 'dog', 'elephant',
            'keyboard', 'knife', 'oven', 'truck']

adv_name = ["pgd", "fgsm", "deepfool", 'onepix']


for dn in target_dataset:
    dataset = load_dataset(dn)
    
    for an in advs_name:
    
        for mn in model_name:
            model = load_model(mn)
            dataset = load_dataset('IN')
            
            
            # AA
            if adv_name == "pgd":
                atk = torchattacks.PGD(model)
            elif adv_name == "fgsm":
                atk = torchattacks.FGSM(model)
            elif adv_name == "deepfool":
                atk = torchattacks.DeepFool(model)
            elif adv_name == "onepix":
                atk = torchattacks.OnePixel(model, pixels=1)
            
            with torch.no_grad():
                proc_dict = {}
                
                logits_arr = np.empty((0, 1000))
                label_arr = []
                for images, target, _ in tqdm(dataset.loader):
                    images = images.to(device())
                    label = [int(classes.index(i)) for i in target]
                    # label = int(classes.index(target[0]))
                    
                    
                    
                    adv_images = atk(images, label)
                    
                    # extract adv's SoftMax
                    logits = out_logits(mn, model, adv_images)
                    logits = logits.reshape((batch_size, -1))
                    
                    logits_arr = np.append(logits_arr, logits, axis=0)
                    label_arr.extend(label)
                
                for ci, cn in enumerate(classes):
                    proc_dict[cn] = logits_arr[np.where(np.array(label_arr) == ci)]
                    print(proc_dict[cn].shape)
                    
                # save softmax
                save_dir = '/home/ueda-tatsuya/デスクトップ/k-Spectrum/model-vs-human/softmax_dict_1017/'
                with open(save_dir + f'{an}_{dn}_to_{mn}.pkl', 'wb') as p:
                    pickle.dump(proc_dict, p)


