import torch
from torch import nn
import torchvision
from PIL import Image
import os
import cv2

def load_image(image):
  PIL_image = Image.fromarray(image)
  out = torchvision.transforms.Resize((256,128))(PIL_image) 
  out_tensor = torchvision.transforms.ToTensor()(out)
  out_tensor = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(out_tensor)
  out_tensor = torch.unsqueeze(out_tensor, dim=0).float()

  return out_tensor