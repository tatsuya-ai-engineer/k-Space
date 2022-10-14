import torch
import numpy as np
from tqdm import tqdm
import pickle

from k_utils.modify import *
from k_utils.utils import *


###
target_dataset = ['IN']
model_name = ['resnet-IN']

classes = ['airplane', 'bear', 'bicycle', 'bird', 
              'boat', 'bottle', 'car', 'cat', 
              'chair', 'clock', 'dog', 'elephant',
              'keyboard', 'knife', 'oven', 'truck']

###

batch_size = 8
for dn in target_dataset:
    dataset = load_dataset(dn, batch_size=batch_size)
    
    for mn in model_name:
        model = load_feature_extractor(mn)
        print(f"{dn} -- {mn}")
        
        
        with torch.no_grad():
            proc_dict = {}
            
            logits_arr = np.empty((0, 401408))
            label_arr = []
            for images, target, _ in tqdm(dataset.loader):
                images = images.to(device())
                label = [int(classes.index(i)) for i in target]
                # label = int(classes.index(target[0]))
                
                logits = out_logits(mn, model, images)
                logits = logits.reshape((batch_size, -1))
                
                logits_arr = np.append(logits_arr, logits, axis=0)
                label_arr.extend(label)
            
            for ci, cn in enumerate(classes):
                proc_dict[cn] = logits_arr[np.where(np.array(label_arr) == ci)]
                print(proc_dict[cn].shape)
                
            # save softmax
            save_dir = '/home/ueda-tatsuya/デスクトップ/k-Spectrum/model-vs-human/softmax_dict_1013/'
            with open(save_dir + f'layer2_{dn}_to_{mn}.pkl', 'wb') as p:
                pickle.dump(proc_dict, p)