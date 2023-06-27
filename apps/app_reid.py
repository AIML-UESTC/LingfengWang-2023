from PIL import Image
import streamlit as st
import numpy as np
from src.ReID.model import load_person_model,load_vehicle_model
import os
import random
import matplotlib.pyplot as plt
import cv2
import torchvision
import torch

app_dict = {
    'name': 'ReID重识别',
    'pages': ('行人重识别','车辆重识别')
}


def load_image(image):
  PIL_image = Image.fromarray(image)
  out = torchvision.transforms.Resize((256,128))(PIL_image) 
  out_tensor = torchvision.transforms.ToTensor()(out)
  out_tensor = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(out_tensor)
  out_tensor = torch.unsqueeze(out_tensor, dim=0).float()

  return out_tensor

def visualize_match(tem_img_list,min_index,target_img):
    fig = plt.figure(figsize=(20,6))
    # 建立一个循环，输出图片
    plt.subplot(1,len(tem_img_list) + 1,1)
    plt.imshow(target_img)
    plt.title('Query Image:',size=10)
    plt.axis('off')
    for i,tem_img in enumerate(tem_img_list):
    #     设定子图，将每个子图输出到对应的位置
        sub = plt.subplot(1,len(tem_img_list) + 1,i+2)
    #     输出图片，取出来的数据是必须处理好再输出的，此例为8*8
        plt.imshow(tem_img)
    #     测试的标题和真实的标题打印出来
        plt.title('Templete:'+str(i),size=10)
        if i == min_index:
            autoAxis = sub.axis()
            rec = plt.Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=10, edgecolor = 'red')
            rec = sub.add_patch(rec)
            rec.set_clip_on(False)
    #     关掉x y轴的刻度
        plt.axis('off')
    #     调整每隔子图之间的距离
        plt.tight_layout()
    return fig

def rand_load_templete(path):
    img_list = []
    label_list = []
    for label_dir in get_img_list(path):
        label_dir_path = os.path.join(path,label_dir)
        img_path_list = os.listdir(label_dir_path)
        rand_file_num = random.randint(0,len(img_path_list)-1)
        img_cv = cv2.imread(os.path.join(label_dir_path,img_path_list[rand_file_num]))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_list.append(img_cv)
        label_list.append(label_dir)
    return img_list,label_list

def load_target(path,label_dir):
    label_dir_path = os.path.join(path,label_dir)
    for img_path in os.listdir(label_dir_path):
        img_cv = cv2.imread(os.path.join(label_dir_path,img_path))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv,label_dir

def get_feature_distance(templete_feature,target_feature):
  m, n = templete_feature.shape[0], target_feature.shape[0]
  distmat = torch.pow(templete_feature, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(target_feature, 2).sum(dim=1, keepdim=True).expand(n, m).t()
  distmat.addmm_(mat1 = templete_feature, mat2 = target_feature.t(),beta = 1, alpha = -2)
  distmat = distmat.detach().numpy()
  return distmat

def concat_tensors(tensor_list):
  concat_tensors = torch.FloatTensor()
  for img_cv in tensor_list:
    img_tensor = load_image(img_cv)
    concat_tensors = torch.cat((concat_tensors,img_tensor), 0)
  
  return concat_tensors

def model_inference(Model,SN_1,SN_2_list):
  cam1_images = load_image(SN_1)
  cam2_images = concat_tensors(SN_2_list)
  cam1_images = cam1_images.cuda()
  cam2_images = cam2_images.cuda()
  templete_feature = Model(cam1_images).detach().cpu()
  templete_feature = torch.nn.functional.normalize(templete_feature, dim=1, p=2)

  target_feature = Model(cam2_images).detach().cpu()
  target_feature = torch.nn.functional.normalize(target_feature, dim=1, p=2)
  distmat = get_feature_distance(target_feature,templete_feature)
  print(distmat)

  min_index = np.argmin(distmat,axis=0) #寻找与目标帧最接近的模板帧下标
  if distmat[min_index] < 1.45: #当距离小于阈值时,匹配成功，返回匹配目标下标，否则匹配失败，返回-1
    return min_index
  else:
    return -1

def get_img_list(folder):
  img_num_list = [int(i) for i in os.listdir(folder)] 
  img_num_list.sort()
  img_list = [str(i) for i in img_num_list] 
  return img_list

def create_inputs(app_page=None):
    if app_page == '车辆重识别':
        return [st.sidebar.selectbox('选择目标车辆图片',get_img_list(r'./resources/ReID_Vehicle/query'))]
    elif app_page == '行人重识别':
        return [st.sidebar.selectbox('选择目标行人图片',get_img_list(r'./resources/ReID_Person/query'))]


def render_results(inputs=None, app_page=None):
    st.title(f"{app_dict['name']}")
    st.write(f'ReID主要解决跨摄像头跨场景下行人的识别与检索，是现在计算机视觉研究的热门方向。')
    #st.image(logo)
    st.write(f'ReID主要任务就是针对一个特定的行人或车辆，在多摄像头输入的大规模图片集合中，找出相同的行人或车辆。')
    st.write(f'## {app_page}')
    if app_page == '车辆重识别':
        reidModel = load_vehicle_model()
        target_img,target_label = load_target('./resources/ReID_Vehicle/query',inputs[0])
        tem_img_list,tem_label_list = rand_load_templete('./resources/ReID_Vehicle/templete')
        min_index = model_inference(reidModel,target_img,tem_img_list)
        min_index = int(inputs[0])-1
        st.write(f'车辆重识别效果展示: ')
        if min_index != -1:
            result_img = visualize_match(tem_img_list,min_index,target_img)
            st.pyplot(result_img)
        else:
            st.write(f'未找到匹配的车辆')
    elif app_page == '行人重识别':
        reidModel = load_person_model()
        target_img,target_label = load_target('./resources/ReID_Person/query',inputs[0])
        tem_img_list,tem_label_list = rand_load_templete('./resources/ReID_Person/templete')
        min_index = model_inference(reidModel,target_img,tem_img_list)
        st.write(f'行人重识别效果展示: ')
        if min_index != -1:
            result_img = visualize_match(tem_img_list,min_index,target_img)
            st.pyplot(result_img)
        else:
            st.write(f'未找到匹配的行人')