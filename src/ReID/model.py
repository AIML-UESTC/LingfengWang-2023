import cv2
import numpy as np
from src.ReID.modeling import build_model
from src.ReID.config import cfg as reidCfg
from src.ReID.util import load_image
import os
import numpy as np
import torch
import torchvision
from PIL import Image
from io import BytesIO
import time
import logging

def load_vehicle_model():
  reidModel = build_model(reidCfg, num_classes=1026)
  reidModel.load_param("./ckpts/ReID_Vehicle/resnet50.pth")
  reidModel.eval().cuda()
  
  return reidModel

def load_person_model():
  reidModel = build_model(reidCfg, num_classes=1026)
  reidModel.load_param("./ckpts/ReID_Person/resnet50.pth")
  reidModel.eval().cuda()
  
  return reidModel