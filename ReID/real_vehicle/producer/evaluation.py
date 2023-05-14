import pika
import json
import os
import base64
import cv2
import time
import sys

receive_cnt = 0
send_num = 0
start_time = 0
receive_match = {}
car_gt_match,car_tem_list = {},{}

import random
import matplotlib.pyplot as plt

def visualize_match(tem_img_list,min_index,target_img,file_name):
    plt.figure(figsize=(20,6))
    # 建立一个循环，输出图片
    plt.subplot(1,len(tem_img_list) + 1,1)
    plt.imshow(target_img)
    plt.axis('off')
    for i,tem_img in enumerate(tem_img_list):
    #     设定子图，将每个子图输出到对应的位置
        sub = plt.subplot(1,len(tem_img_list) + 1,i+2)
    #     输出图片，取出来的数据是必须处理好再输出的，此例为8*8
        plt.imshow(tem_img)
    #     测试的标题和真实的标题打印出来
        plt.title('Templete:'+str(i),size=5)
        if i == min_index:
            autoAxis = sub.axis()
            rec = plt.Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=10, edgecolor = 'red')
            rec = sub.add_patch(rec)
            rec.set_clip_on(False)
    #     关掉x y轴的刻度
        plt.axis('off')
    #     调整每隔子图之间的距离
        plt.tight_layout()
    plt.savefig(file_name) 

def rand_load_templete(path):
    img_list = []
    label_list = []
    img_name_list = []
    for label_dir in os.listdir(path):
        label_dir_path = os.path.join(path,label_dir)
        img_path_list = os.listdir(label_dir_path)
        rand_file_num = random.randint(0,len(img_path_list)-1)
        img_cv = cv2.imread(os.path.join(label_dir_path,img_path_list[rand_file_num]))
        img_name_list.append(img_path_list[rand_file_num])
        img_list.append(img_cv)
        label_list.append(label_dir)
    return img_list,label_list,img_name_list

def load_target(path):
    img_list = []
    label_list = []
    img_name_list = []
    for label_dir in os.listdir(path):
        label_dir_path = os.path.join(path,label_dir)
        for img_path in os.listdir(label_dir_path):
            img_cv = cv2.imread(os.path.join(label_dir_path,img_path))
            img_name_list.append(img_path)
            img_list.append(img_cv)
            label_list.append(label_dir)
    return img_list,label_list,img_name_list

def pack_json(camid1,camid2,trkid1,trkid2,image1,image2):
  #camid1(string) camid2(list) trkid1(string) trktem(list) image1(base64 string) image2(list)
  pack_dict = {}
  pack_dict['C_Test_id'] = camid1
  pack_dict['C_Tem_id'] = camid2
  pack_dict['C_Test_trkid'] = trkid1
  pack_dict['C_Tem_trkid'] = trkid2
  pack_dict['C_Test_img'] = image1
  pack_dict['C_Tem_img'] = image2
  return json.dumps(pack_dict)

def parse_json(receive_dict):
  return receive_dict['C_Test_id'],receive_dict['C_Test_trkid'],receive_dict['C_Tem_id'],receive_dict['C_Tem_trkid']

def generate_msgs(type):
  #加载待匹配图像，并采用base64编码
  #加载SN1图像
  messages = []
  gt_match = {}
  generate_tem_list = {}
  target_img_list,target_label_list,target_img_name_list = load_target(os.path.join('./dataset/query',type))
  
  for target_img_cv,target_label,target_img_name in zip(target_img_list,target_label_list,target_img_name_list):
    image = cv2.imencode('.jpg',target_img_cv)[1]
    image_data = str(base64.b64encode(image))[2:-1]
    trkid1 = target_img_name[:-4]
    camid1 = '1'
    SN_1 = image_data

    #遍历先前帧文件夹,加载先前帧图像SN2 
    trkid2_list = []
    camid2_list = []
    SN_2_list = []

    match_trkid = None
    tem_img_list,tem_label_list,tem_img_name_list = rand_load_templete(os.path.join('./dataset/templete',type))
    for tem_img_cv,tem_label,tem_img_name in zip(tem_img_list,tem_label_list,tem_img_name_list):
      image = cv2.imencode('.jpg',tem_img_cv)[1]
      image_data = str(base64.b64encode(image))[2:-1]
      trkid = tem_img_name[:-4]
      camid = '2'
      trkid2_list.append(trkid)
      camid2_list.append(camid)
      SN_2_list.append(image_data)
      if target_label == tem_label:
        match_trkid = trkid


    gt_match[trkid1] = match_trkid
    generate_tem_list[trkid1] = trkid2_list
    message=pack_json(camid1,camid2_list,trkid1,trkid2_list,SN_1,SN_2_list)
    messages.append(message)
  return messages, gt_match, generate_tem_list

  # 可视化匹配结果，并保存到本地
def visualize_precess(receive_match):
  global car_gt_match,car_tem_list
  target_trkid_dict = {}
  tem_trkid_dict = {}

  for root, ds, fs in os.walk('./dataset/query'):
    for f in fs:
        fullname = os.path.join(root,f)
        target_trkid_dict[f[:-4]] = fullname
  
  for root, ds, fs in os.walk('./dataset/templete'):
    for f in fs:
        fullname = os.path.join(root,f)
        tem_trkid_dict[f[:-4]] = fullname

  corrects = 0
  cnt = 0

  
  #可视化真实行人图片REID结果
  car_gt_match,car_tem_list
  for trkid1 in car_tem_list.keys():
    cnt = cnt + 1
    if receive_match[trkid1] == car_gt_match[trkid1]:
      corrects = corrects + 1
    target_img = cv2.imread(target_trkid_dict[trkid1])
    tem_img_list = []

    min_index = None
    for idx,car_tem_trkid in enumerate(car_tem_list[trkid1]):
      if car_tem_trkid == receive_match[trkid1]:
        min_index = idx
      tem_img_list.append(cv2.imread(tem_trkid_dict[car_tem_trkid]))
    visualize_match(tem_img_list,min_index,target_img,os.path.join('./match_result',trkid1+'_'+receive_match[trkid1]+'.png'))
  print('Accuracy: ',corrects/cnt)


def callback(ch, method, properties, body):
  cam_id1,trkid1,out_cam_id2_list,out_trkid2_list = parse_json(json.loads(body))
  match_trkid = None
  for trkid2 in out_trkid2_list:
    if trkid2 != '-1':
      match_trkid = trkid2

  global receive_cnt,send_num,start_time,receive_match,generate_tem_list
  ch.basic_ack(delivery_tag = method.delivery_tag)
  receive_cnt = receive_cnt + 1
  receive_match[trkid1] = match_trkid
  print("\r", end="")
  print("receive: {} result".format(receive_cnt), end="")
  if receive_cnt == send_num:
    time_cost = time.time() - start_time
    print("\r", end="")
    print('*******************************')
    print('Receive all %s ReID result' %receive_cnt)
    print('Total time consuming: '+str(time_cost)+' ,FPS:'+str(1.0/(time_cost/receive_cnt)))
    visualize_precess(receive_match)
    
  elif receive_cnt < send_num:
    sys.stdout.flush()
      
def set_environ(): #windows下测试用函数，设置环境变量
  os.environ['mq_account']='reid'
  os.environ['mq_pwd']='reid'
  os.environ['mq_ip']='192.168.101.151'
  os.environ['mq_port']= '5672'

if __name__ == "__main__":
  set_environ()
  mq_account = os.environ.get('mq_account')
  mq_pwd = os.environ.get('mq_pwd')
  mq_ip = os.environ.get('mq_ip')
  mq_port = os.environ.get('mq_port')
  print('Get rabbitmq config from environ: ',mq_account,mq_pwd,mq_ip,mq_port)
  credentials = pika.PlainCredentials(mq_account, mq_pwd)
  connection = pika.BlockingConnection(pika.ConnectionParameters(host = mq_ip,port = mq_port, virtual_host = '/',credentials = credentials))
  channel=connection.channel()
  # 申明消息队列，消息在这个队列传递，如果不存在，则创建队列
  channel.queue_declare(queue = 'Coll2Reid', durable = True) #消息类型Coll2Reid，发送给ReID 
  channel.queue_declare(queue = 'Reid2Coll', durable = True) #消息类型Reid2Coll，返回ReID结果
  # 发送真实车辆测试数据
  messages,car_gt_match,car_tem_list = generate_msgs('real_car')
  for message in messages:
    channel.basic_publish(exchange = '',routing_key = 'Coll2Reid',body = message)
  send_num = send_num + len(messages)
  print('Sending car: ',len(messages))

  print('successfully send %s message to ReID consumer' %send_num)
  print('*******************************')
  start_time = time.time()

  # 告诉rabbitmq，用callback来接收ReID consumer返回消息
  channel.basic_consume('Reid2Coll',callback)
  # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理
  print('Ready to receive result from ReID consumer')
  channel.start_consuming()
  
