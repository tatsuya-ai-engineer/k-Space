###

# You need 
#
# git clone https://github.com/bethgelab/model-vs-human.git

###




import torch
import numpy as np
from tqdm import tqdm
import pickle

from k_utils.modify import *
from k_utils.utils import *

###
target_dataset = ['IN', 'SIN']
model_name = ['resnet-IN', 'resnet-SIN', 'vit-IN']

classes = ['airplane', 'bear', 'bicycle', 'bird', 
              'boat', 'bottle', 'car', 'cat', 
              'chair', 'clock', 'dog', 'elephant',
              'keyboard', 'knife', 'oven', 'truck']

proc_name = ['crop', 'rotate', 'noise']
###

for pn in proc_name:

    for dn in target_dataset:
        dataset = load_dataset(dn)
        
        for mn in model_name:
            model = load_model(mn)
            print(f"{pn} -- {dn} -- {mn}")
            
            proc_dict = {}
            
            with torch.no_grad():
                for ci, cn in enumerate(classes):
                    print(cn)
                    proc_list = []
                    
                    for data_counter, (images, target, _) in enumerate(dataset.loader):
                        # images = images.to(device())
                        label = int(classes.index(target[0]))
                        
                        if ci == label:
                            if pn == "crop":
                                proc_X = center_cropped_dataset(images[0])
                            elif pn == "rotate":
                                proc_X = rotated_dataset(images[0])
                            elif pn == "noise":
                                    proc_X = noised_dataset(images[0])
                                
                            images = proc_X.to(device())
                            logits = out_logits(mn, model, images)      # (p, 1000)
                            
                            proc_list.append(logits)
                    
                    print(len(proc_list))
                    proc_dict[cn] = proc_list
                    
                # save softmax
                save_dir = '/home/ueda-tatsuya/デスクトップ/k-Spectrum/model-vs-human/softmax_dict_1013/'
                with open(save_dir + f'{pn}{dn}_to_{mn}.pkl', 'wb') as p:
                    pickle.dump(proc_dict, p)