from PIL import Image
#from src.Face.mtcnn import detect_faces, show_bboxes
from src.Face.centerface.inference import load_model, detect_faces, draw
import streamlit as st
import cv2
import os
from io import BytesIO
import numpy as np 

app_dict = {
    'name': '人脸检测与对齐',
    'pages': ('本地图片', '上传图片')
}

#logo = Image.open('./assets/app_a_litt_up.jpg')

def run_face_detection(model,img):
    bounding_boxes, landmarks = detect_faces(model,img)
    img = draw(img, bounding_boxes, landmarks)
    st.image(img)
    

def create_inputs(app_page=None):
    if app_page == '本地图片':
         return [st.sidebar.selectbox('选择目标图片',os.listdir(r'./resources/Face_Detection'))]
    elif app_page == '上传图片':
        return [st.file_uploader("选择上传文件")]


def render_results(inputs=None, app_page=None):
    st.title(f"{app_dict['name']}")
    st.write(f'## {app_page}')
    st.write(f'检测图片人脸与面部关键点')

    model = load_model()
    
    if app_page == '本地图片':
        img_path = os.path.join(r'./resources/Face_Detection',inputs[0])
        img = Image.open(img_path)
        img = np.asarray(img)
        run_face_detection(model,img)
    elif app_page == '上传图片':
        if inputs[0] is not None:
            # To read file as bytes:
            bytes_data = inputs[0].getvalue()
            #将字节数据转化成字节流
            bytes_data = BytesIO(bytes_data)
            #Image.open()可以读字节流
            img = Image.open(bytes_data)
            img = np.asarray(img)
            run_face_detection(model,img)
