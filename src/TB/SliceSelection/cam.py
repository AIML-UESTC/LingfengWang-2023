# ----------------------------------------------------------------------------------------------------------
# TS-CAM
# Copyright (c) Learning and Machine Perception Lab (LAMP), SECE, University of Chinese Academy of Science.
# ----------------------------------------------------------------------------------------------------------
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
import xlwt
import math
import shutil
from ActivationMap import GradCam

def transform_image(img_path,mask_array):
    image = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(image)[0, :, :]
    img[np.where(mask_array == 0)] = -1024
    img_max = np.max(img)
    img_min = np.min(img)
    img_range = img_max - img_min
    img = (img.astype('float32') - img_min) / img_range
    img = (img*255).astype('uint8')
    img = np.stack((img,img,img),axis=2)
    PIL_image = Image.fromarray(img)
    out = torchvision.transforms.Resize((224, 224))(PIL_image)
    out_tensor = torchvision.transforms.ToTensor()(out)
    return out_tensor

def show_cam_on_image(img, mask_pos, file_name = 'CAM.jpg'):
    mask_pos = mask_pos/(mask_pos.max()-mask_pos.min())
    pos_heatmap = cv2.applyColorMap(np.uint8(255 * mask_pos), cv2.COLORMAP_JET)
    pos_heatmap = np.float32(pos_heatmap) / 255

    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    pos_cam = pos_heatmap + np.float32(img)
    pos_cam = pos_cam / np.max(pos_cam)

    squeeze_img = np.squeeze(img)

    plt.figure()
    plt.subplot(121)
    plt.imshow(squeeze_img, cmap='gray')
    plt.subplot(122)
    pos_cam = cv2.cvtColor(pos_cam, cv2.COLOR_BGR2RGB)
    plt.imshow(pos_cam)
    plt.savefig(file_name)
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output2")
    parser.add_argument('--resume', type=str, default="./checkpoint/model.pth.tar", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    model = getattr(ssenet_resnet, "SSe_resnet_18")(num_classes=2)
    print(model)
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        torch.backends.cudnn.enabled = False
        #model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])
    
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                target_layer_names=["1"], use_cuda=True)

    dataset_folder = r'L:\Hainan-test'
    result_folder = 'output'
    img_save_root_path = os.path.join(result_folder,'ActivationMap')
    target_folder = os.path.join(result_folder,'SelectedSlices')
    
    thresh_area = 800
    thresh_heat =0.5

    with torch.no_grad():
        for idx,patient_folder in enumerate(os.listdir(dataset_folder)):
            patient_path = os.path.join(dataset_folder, patient_folder)
            img_save_patient_path = os.path.join(img_save_root_path, patient_folder)
            if not os.path.exists(img_save_patient_path):
                os.makedirs(img_save_patient_path)
            patient_voxel_volume = [0,0,0,0,0]
            patient_cam_area = [0,0,0,0,0]
            for file in os.listdir(patient_path):
                if file.endswith('dcm'):
                    #read dcm slice and corresponding lung mask
                    img_path = os.path.join(patient_path, file)
                    print('Processing ', img_path)
                    mask_path = os.path.join(patient_path,'lobes_' + file.replace('dcm','nii.gz'))
                    mask_image = sitk.ReadImage(mask_path)
                    mask_array = sitk.GetArrayFromImage(mask_image)[0, :, :]
                    out_tensor = transform_image(img_path,mask_array)

                    #activation map inference 
                    image_variable = Variable(torch.unsqueeze(out_tensor, dim=0).float(), requires_grad=False)
                    mask_pos, class_output, prob_output = grad_cam(image_variable, 1)
                    #Save Activation Map result
                    if class_output == 1:
                        save_img_path = os.path.join(img_save_patient_path,file[:-4] + '_' + str(class_output) + '.jpg')
                        print('saving ', save_img_path)
                        show_cam_on_image(out_tensor, mask_pos, save_img_path)

if __name__ == "__main__":
    main()
