from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from torch.autograd import Function
import ssenet_resnet
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
from PIL import Image 
import SimpleITK as sitk
import math
from src.TB.SliceSelection.ActivationMap import GradCam

def transform_image(img_path,mask_array):
    image = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(image)[0, :, :]
    img[np.where(mask_array == 0)] = -1024
    img_max = np.max(img)
    img_min = np.min(img)
    img_range = img_max - img_min
    img = (img.astype('float32') - img_min) / img_range
    PIL_image = Image.fromarray(img)
    out = torchvision.transforms.Resize((224, 224))(PIL_image)
    out_tensor = torchvision.transforms.ToTensor()(out)
    return out_tensor

def show_cam_on_image(img, mask_array,mask_pos, file_name = './assets/TB_grad_cam.jpg'):
    pos_heatmap = cv2.applyColorMap(np.uint8(255 * mask_pos), cv2.COLORMAP_JET)
    pos_heatmap = np.float32(pos_heatmap) / 255

    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    pos_cam = pos_heatmap + np.float32(img)
    pos_cam = pos_cam / np.max(pos_cam)

    squeeze_img = np.squeeze(img)

    plt.figure()
    plt.subplot(131)
    plt.imshow(squeeze_img, cmap='gray')
    plt.subplot(132)
    plt.imshow(mask_array, cmap='gray')
    plt.subplot(133)
    pos_cam = cv2.cvtColor(pos_cam, cv2.COLOR_BGR2RGB)
    plt.imshow(pos_cam)
    plt.savefig(file_name)
    plt.close('all')

def get_slice_model():
    model = getattr(ssenet_resnet, "SSe_resnet_18")(num_classes=2)
    checkpoint = torch.load(r'./ckpts/TB/slice_selection.pth.tar')
    base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
    model.load_state_dict(base_dict)
    model = model.cuda()
    torch.backends.cudnn.enabled = False    
    grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["1"], use_cuda=True)
    return grad_cam

def grad_cam_inference(grad_cam,img_path):
    #activation map inference 
    base_name = os.path.basename(img_path)
    mask_path = os.path.join('./resources/LungSeg','lobes_' + base_name.replace('dcm','nii.gz'))
    print(mask_path)
    mask_image = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image)[0, :, :]
    out_tensor = transform_image(img_path,mask_array)
    image_variable = Variable(torch.unsqueeze(out_tensor, dim=0).float(), requires_grad=False)
    mask_pos, class_output, prob_output = grad_cam(image_variable, 1)
    show_cam_on_image(out_tensor, mask_array, mask_pos, './assets/TB_grad_cam.jpg')
    return class_output

                