import pika
import json
import base64
import cv2
import numpy as np
from reid.modeling import build_model
from reid.config import cfg as reidCfg
from reid.util import load_image
import os
import numpy as np
import torch
import torchvision
from PIL import Image
from io import BytesIO
import time
import logging

visualization = True

def base64_to_image(img_base64):
  img_data = base64.b64decode(img_base64)
  img_array = np.fromstring(img_data,np.uint8)
  img = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
  return img

def parse_json(receive_dict):
  img_cv = base64_to_image(receive_dict['C_Test_img'])
  SN_1 = img_cv

  SN_2_list = []
  for img_base64 in receive_dict['C_Tem_img']:
    img_cv = base64_to_image(img_base64)
    SN_2_list.append(img_cv)
  return receive_dict['C_Test_id'],receive_dict['C_Tem_id'],receive_dict['C_Test_trkid'],receive_dict['C_Tem_trkid'],SN_1,SN_2_list

def pack_json(camid1,camid2,trkid1,trkid2):
  pack_dict = {}
  pack_dict['C_Test_id'] = camid1
  pack_dict['C_Tem_id'] = camid2
  pack_dict['C_Test_trkid'] = trkid1
  pack_dict['C_Tem_trkid'] = trkid2
  return json.dumps(pack_dict)

def get_feature_distance(templete_feature,target_feature):
  m, n = templete_feature.shape[0], target_feature.shape[0]
  distmat = torch.pow(templete_feature, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(target_feature, 2).sum(dim=1, keepdim=True).expand(n, m).t()
  distmat.addmm_(mat1 = templete_feature, mat2 = target_feature.t(),beta = 1, alpha = -2)
  distmat = distmat.detach().numpy()  # <class 'tuple'>: (3, 12)
  return distmat

def concat_tensors(tensor_list):
  concat_tensors = torch.FloatTensor()
  for img_cv in tensor_list:
    img_tensor = load_image(img_cv)
    concat_tensors = torch.cat((concat_tensors,img_tensor), 0)
  
  return concat_tensors

def draw_match_result(save_path,cam_id1,out_cam_id2,trkid1,out_trkid2,img1,img2):
  save_name = 'C'+str(cam_id1)+'T'+str(trkid1)+'C'+str(out_cam_id2)+'T'+str(out_trkid2)
  save_img_path = os.path.join(save_path,save_name)
  os.makedirs(save_img_path,exist_ok=True)
  img1 = Image.fromarray(img1)
  img2 = Image.fromarray(img2)
  img1.save(os.path.join(save_img_path,'query.jpg'))
  img2.save(os.path.join(save_img_path,'target.jpg'))

# 定义一个回调函数来处理消息队列中的消息
def callback(ch, method, properties, body):
  global cuda,logging
  start_time = time.time()
  cam_id1,cam_id2_list,trkid1,trkid2_list,SN_1,SN_2_list = parse_json(json.loads(body))
  cam_id1 = str(cam_id1)
  trkid1 = str(trkid1)
  print('Receive Test frame from camid: ',cam_id1,' ,track: ',trkid1)
  print('Receive Temp frame from camid: ',cam_id2_list,' ,track: ',trkid2_list)
  cam1_images = load_image(SN_1)
  cam2_images = concat_tensors(SN_2_list)
  if cuda:
    cam1_images = cam1_images.cuda()
    cam2_images = cam2_images.cuda()
  templete_feature = reidModel(cam1_images).detach().cpu()
  templete_feature = torch.nn.functional.normalize(templete_feature, dim=1, p=2)

  target_feature = reidModel(cam2_images).detach().cpu()
  target_feature = torch.nn.functional.normalize(target_feature, dim=1, p=2)
  distmat = get_feature_distance(target_feature,templete_feature)
  print(distmat)

  out_cam_id2_list = ['-1']*distmat.shape[0]
  out_trkid2_list = ['-1']*distmat.shape[0]
  min_index = np.argmin(distmat,axis=0) #寻找与目标帧最接近的模板帧下标
  if distmat[min_index] < 1.45: #当距离小于阈值时,匹配成功，返回匹配目标下标，否则匹配失败，返回-1
    match_camid = cam_id2_list[min_index[0]]
    match_trkid = trkid2_list[min_index[0]]
    out_cam_id2_list[min_index[0]] = match_camid
    out_trkid2_list[min_index[0]] = match_trkid
    logging.info(f'Cam {cam_id1} Track {trkid1} match to Cam {match_camid} Track {match_trkid}')
    if visualization == 'True':
      draw_match_result('./match_result',cam_id1,match_camid,trkid1,match_trkid,SN_1,SN_2_list[min_index[0]])
  else:
    logging.info(f'Cam {cam_id1} Track {trkid1} match failed, no matched tracklet')
  print(out_cam_id2_list)
  print(out_trkid2_list)
  channel.basic_publish(exchange = '',routing_key = 'Reid2Coll',body = pack_json(cam_id1,out_cam_id2_list,trkid1,out_trkid2_list))
  print('ReID result send to Reid2Coll')
  print('time: ',time.time()-start_time)
  ch.basic_ack(delivery_tag = method.delivery_tag)

def set_environ(): ##windows下测试用函数，设置环境变量
  os.environ['mq_account']='reid'
  os.environ['mq_pwd']='reid'
  os.environ['mq_ip']='192.168.101.151'
  os.environ['mq_port']= '5672'
  os.environ['flag_vis']= 'True'
    
if __name__ == "__main__":
  #实例化reid模块
  logging.basicConfig(filename=os.path.join('./match_result',"log.txt"), level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
  logging.getLogger()
  if torch.cuda.is_available():
    cuda = True
  else:
    cuda = False
  reidModel = build_model(reidCfg, num_classes=1026)
  reidModel.load_param(r"resnet50_pedestrian.pth")
  if cuda:
    reidModel.eval().cuda()
  else:
    reidModel.eval()

  set_environ()
  mq_account = os.environ.get('mq_account')
  mq_pwd = os.environ.get('mq_pwd')
  mq_ip = os.environ.get('mq_ip')
  mq_port = os.environ.get('mq_port')
  visualization = os.environ.get('flag_vis','True') #default True
  print('Get rabbitmq config from environ: ',mq_account,mq_pwd,mq_ip,mq_port)
  credentials = pika.PlainCredentials(mq_account, mq_pwd)
  connection = pika.BlockingConnection(pika.ConnectionParameters(host = mq_ip,port = mq_port, virtual_host = '/',credentials = credentials))
  channel = connection.channel()
  # 申明消息队列，消息在这个队列传递，如果不存在，则创建队列
  channel.queue_declare(queue = 'Coll2Reid', durable = True) #消息类型Coll2Reid，发送给ReID 
  channel.queue_declare(queue = 'Reid2Coll', durable = True) #消息类型Reid2Coll，返回ReID结果

  # 告诉rabbitmq，用callback来接收消息
  channel.basic_qos(prefetch_count=1)
  channel.basic_consume('Coll2Reid',callback)
  # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理
  print('Successfully connect to rabbitmq')
  channel.start_consuming()
