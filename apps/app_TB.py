from PIL import Image
import streamlit as st
from src.TB.Detector.Detector import inference,get_detector_model
from src.TB.Classifier.classifier import get_classifier_model
from src.TB.SliceSelection.inference import get_slice_model,grad_cam_inference
import os
import SimpleITK as sitk

Focus_classes=['Calcified granulomas','Cavitation','Centrilobular and tree-in-bud nodules','Clusters of nodules','Consolidation','Fibronodular scarring']

app_dict = {
    'name': '肺结核全流程诊断系统',
    'pages': ('感染范围评估', '病灶检测识别')
}


def create_inputs(app_page=None):
    if app_page == '感染范围评估':
        return [st.sidebar.selectbox('选择DICOM影像',os.listdir(r'./resources/TB'))]
    elif app_page == '病灶检测识别':
        return [st.sidebar.selectbox('选择DICOM影像',os.listdir(r'./resources/TB'))]


def render_results(inputs=None, app_page=None):
    st.title(f"{app_dict['name']}")
    st.write(f'## {app_page}')
    if app_page == '感染范围评估':
        slice_model = get_slice_model()
        img_path = os.path.join(r'./resources/TB',inputs[0])
        class_output = grad_cam_inference(slice_model,img_path)
        st.write(f'左侧为影像原图，中间为肺叶分割结果，右侧为预估感染范围: ')
        st.image('./assets/TB_grad_cam.jpg')
    elif app_page == '病灶检测识别':
        classifier_model = get_classifier_model()
        detect_model = get_detector_model()
        img_path = os.path.join(r'./resources/TB',inputs[0])
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))[0,:,:]
        inference(detect_model,classifier_model,image)
        result_img = Image.open('./assets/TB_result.jpg')
        st.write(f'对当前层面病灶进行检测，同时对各病灶进行分类: ')
        st.image(result_img)

