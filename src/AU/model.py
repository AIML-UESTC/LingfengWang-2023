import torch
import numpy as np
import os
import time
from src.AU.sformer import SpatialFormer
import torch.backends.cudnn as cudnn
from torchvision import transforms
import cv2
from PIL import Image


def au_to_str(arr):
    str = "{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}".format(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], arr[10], arr[11])
    return str

def ex_to_str(arr):
    str = "{:d}".format(arr)
    return str

def va_to_str(v,a):
    str = "{:.3f},{:.3f}".format(v, a)
    return str

def get_sformer():
    cudnn.enabled = True
    device = torch.device("cuda")
    # model
    model = SpatialFormer()
    model = model.to(device)
    model.load_state_dict(torch.load(r'./ckpts/AU/sformer.pth'))
    model.eval()
    # disable grad, set to eval
    for p in model.parameters():
        p.requires_grad = False
    for p in model.children():
        p.train(False)
    
    return model

def predict_au(model,img):
    img.cuda()
    predict = model(img)
    logits_au = predict[:, :12]
    pred_au = torch.sigmoid(logits_au).detach().cpu().squeeze().numpy()
    round_au = np.round(pred_au).astype(np.int)

    return round_au

def predict_ex(model,img):
    img.cuda()
    predict = model(img)
    logits_ex = predict[:, 12:20]
    pred_ex = torch.argmax(logits_ex, dim=1).detach().cpu().numpy().reshape(-1)
    return pred_ex

def process_img(img):
    transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    image = Image.fromarray(img)
    image = transform(image)
    image = image.unsqueeze(0).cuda()
    return image