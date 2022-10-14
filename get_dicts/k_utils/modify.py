
import numpy as np

import torch
from torchvision import transforms

from k_utils.utils import device

###################
def rotated_dataset(target_img: np.ndarray, angles=180) -> torch.tensor:
  # target_img : each pixel is in [0, 1]

#   target_img = torch.from_numpy(target_img.astype(np.float32)).clone()
#   target_img = target_img.permute((2, 0, 1))
  
  imgs = torch.empty(0,3,target_img.shape[1],target_img.shape[1])
  for angle in range(0, angles, 10):
    rotater = transforms.RandomRotation(degrees=(angle, angle))
    rotated_imgs = rotater(target_img)

    imgs = torch.cat([imgs, rotated_imgs.unsqueeze(dim=0)], dim=0)
  return imgs

###################
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def noised_dataset(target_img: np.ndarray) -> torch.tensor:
  # target_img : each pixel is in [0, 1]

#   target_img = torch.from_numpy(target_img.astype(np.float32)).clone()
#   target_img = target_img.permute((2, 0, 1))
  
  imgs = torch.empty(0,3,target_img.shape[1],target_img.shape[1])
  for angle in range(11):
    rotater =  AddGaussianNoise(0., angle*0.1)
    rotated_imgs = rotater(target_img)

    imgs = torch.cat([imgs, rotated_imgs.unsqueeze(dim=0)], dim=0)
  return imgs

###################
def saturated_dataset(target_img: np.ndarray) -> torch.tensor:
  # target_img : each pixel is in [0, 1]

#   target_img = torch.from_numpy(target_img.astype(np.float32)).clone()
#   target_img = target_img.permute((2, 0, 1))
  
  imgs = torch.empty(0,3,target_img.shape[1],target_img.shape[1])
  for factor in range(0, 21, 1):
    rotated_imgs = transforms.functional.adjust_saturation(target_img, factor*0.1)

    imgs = torch.cat([imgs, rotated_imgs.unsqueeze(dim=0)], dim=0)
  return imgs

###################
def center_cropped_dataset(target_img: np.ndarray) -> torch.tensor:
  # target_img : each pixel is in [0, 1]

#   target_img = torch.from_numpy(target_img.astype(np.float32)).clone()
#   target_img = target_img.permute((2, 0, 1))

  target_size = target_img.shape[1]

  imgs = torch.empty(0,3,target_img.shape[1],target_img.shape[1])
  for scale in range(1, target_size+1, 10):
    cropper = transforms.Compose(
        [
        transforms.CenterCrop(size=scale),
        transforms.Resize(target_size)
        ]
    )
    cropped_img = cropper(target_img)

    imgs = torch.cat([imgs, cropped_img.unsqueeze(dim=0)], dim=0)
  return imgs

