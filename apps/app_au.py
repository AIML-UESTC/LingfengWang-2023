from PIL import Image
import streamlit as st
import os
import time
import cv2
from PIL import Image
import numpy as np
import io
from src.AU.model import process_img,predict_au,predict_ex,get_sformer

app_dict = {
    'name': '人脸情绪行为分析',
    'pages': ('面部动作单元','表情分类')
}

AU_dict = {'AU1':'抬起眉毛内角', 'AU2':'抬起眉毛外角', 'AU4':'皱眉',  'AU6':'脸颊提升', 'AU7':'眼睑收紧','AU10':'上唇抬起', 'AU12':'拉动嘴角', 'AU15':'嘴角向下', 'AU23':'收紧嘴唇', 'AU24':'嘴唇按压', 'AU25':'双唇分开', 'AU26':'下巴下降'}
EX_dict = {0:'Neutral',1:'Anger',2:'Disgust',3:'Fear',4:'Happiness',5:'Sadness',6:'Surprise',7:'Other'}


def create_inputs(app_page=None):
    if app_page == '面部动作单元':
        return [st.sidebar.selectbox('选择测试视频',os.listdir(r'./resources/AU/video'))]
    if app_page == '表情分类':
        return [st.sidebar.selectbox('选择测试视频',os.listdir(r'./resources/AU/video'))]


def render_results(inputs=None, app_page=None):
    st.title(f"{app_dict['name']}")
    st.write(f'## {app_page}')
    video_path = os.path.join(r'./resources/AU/video',inputs[0])
    crop_dir = os.path.join(r'./resources/AU/cropped_aligned',inputs[0].split('.')[0])
    cap = cv2.VideoCapture(video_path)  # opencv打开本地视频文件

    if (cap.isOpened() == False):
        st.write("Error opening video stream or file")

    image_placeholder = st.empty()  # 创建空白块使得展示原图
    result_place = st.empty()  # 创建空白块使得展示检测到的人脸
    predict_place = st.empty()  # 创建空白块展示检测到的人脸情绪行为
    frame_cnt = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        if success:
            # 读取当前帧图片数据
            to_show = cv2.resize(frame,(640,360))

            frame_path = os.path.join(crop_dir,str(frame_cnt).zfill(5)+'.jpg')
            if os.path.exists(frame_path):
                cropped = cv2.imread(frame_path)
                img = process_img(cropped)
                model = get_sformer()

                if app_page == '面部动作单元':
                    current_aus = []
                    au_label = predict_au(model,img)
                    flag = False
                    au_str = ''
                    for idx,au_name in enumerate(['AU1', 'AU2', 'AU4', 'AU6','AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']):
                        if au_label[idx] == 1:
                            current_aus.append(au_name)
                            au_str = au_str + au_name + AU_dict[au_name] + '  '
                            flag = True
                    if len(au_str) != 0:
                        predict_place.write('当前帧存在面部动作单元: '+au_str)
                    else:
                        predict_place.write('当前帧未检测到面部动作单元')

                elif app_page == '表情分类':
                    ex_label = predict_ex(model,img)[0]
                    predict_place.write('表情分类: '+EX_dict[ex_label])

                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                result_place.image(cropped, caption='检测对齐人脸')  # 结果图片 将图片帧展示在同一位置得到视频效果
                
            to_show = cv2.cvtColor(to_show, cv2.COLOR_BGR2RGB)
            image_placeholder.image(to_show, caption='原始视频')  # 原图片 将图片帧展示在同一位置得到视频效果
            
            frame_cnt += 1

        else:
            break
    cap.release()

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

