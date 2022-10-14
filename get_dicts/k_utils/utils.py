
import numpy as np

import torch
from torchvision import transforms

from modelvshuman import datasets

from modelvshuman.models.pytorch.model_zoo import model_pytorch
from modelvshuman.models.pytorch.model_zoo import resnet50_trained_on_SIN
from pytorch_pretrained_vit import ViT

def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(data_name):
    if data_name == 'IN':
        dataset = datasets.colour(batch_size=1, num_workers=4)
    elif data_name == 'SIN':
        dataset = datasets.stylized(batch_size=1, num_workers=4)
    return dataset

def load_model(model_name):
    if model_name == 'resnet-IN':
        model = model_pytorch("resnet50")
    elif model_name == 'resnet-SIN':
        model = resnet50_trained_on_SIN("resnet50_trained_on_SIN")
    elif model_name == 'vit-IN':
        model = ViT('B_16_imagenet1k', pretrained=True).to(device())
        
    return model

def out_logits(model_name, model, images):
    if "resnet" in model_name:
        logits = model.forward_batch(images)
    else:
        images = transforms.Compose([
            transforms.Resize((384, 384)), 
            # transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            ])(images).to(device())
        logits = model(images).to('cpu').detach().numpy().copy()
    return logits