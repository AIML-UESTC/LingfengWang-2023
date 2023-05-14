import pika
import json
import os
import base64
import cv2
import argparse
import time
import sys

receive_cnt = 0
send_num = 0
start_time = 0

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
  print(receive_dict['C_Test_id'],receive_dict['C_Test_trkid'])
  print(receive_dict['C_Tem_id'],receive_dict['C_Tem_trkid'])

def generate_msg(image_path):
  #加载待匹配图像，并采用base64编码
  #加载模板帧图像
  templete_path = os.path.join(image_path,'templete')
  file_name = os.listdir(templete_path)[0]
  image_cv = cv2.imread(os.path.join(templete_path,file_name)) #只取一张模板图片
  image = cv2.imencode('.jpg',image_cv)[1]
  image_data = str(base64.b64encode(image))[2:-1]
  trkid1 = file_name.split('_')[0]
  camid1 = file_name.split('_')[1][:-4]
  SN_1 = image_data

  #遍历目标帧文件夹,加载目标帧图像 
  trkid2_list = []
  camid2_list = []
  SN_2_list = []

  target_path = os.path.join(image_path,'target')
  for index,file_name in enumerate(os.listdir(target_path)):
    image_cv = cv2.imread(os.path.join(target_path,file_name))
    image = cv2.imencode('.jpg',image_cv)[1]
    image_data = str(base64.b64encode(image))[2:-1]
    trkid = file_name.split('_')[0]
    camid = file_name.split('_')[1][:-4]
    trkid2_list.append(trkid)
    camid2_list.append(camid)
    SN_2_list.append(image_data)

  message=pack_json(camid1,camid2_list,trkid1,trkid2_list,SN_1,SN_2_list)
  return message
#camid1(string) camid2(list) trkid1(string) trkid2(list) image1(string) image2(list)

  # 定义一个回调函数来处理消息队列中的消息，这里是打印出来
def callback(ch, method, properties, body):
  global receive_cnt,send_num,start_time
  ch.basic_ack(delivery_tag = method.delivery_tag)
  #print('receive ReID result')
  receive_cnt = receive_cnt + 1

  print("\r", end="")
  print("receive: {} result".format(receive_cnt), end="")
  if receive_cnt == send_num:
    time_cost = time.time() - start_time
    print("\r", end="")
    print('*******************************')
    print('Receive all %s ReID result' %receive_cnt)
    print('Total time consuming: '+str(time_cost)+' ,FPS:'+str(1.0/(time_cost/receive_cnt)))
  elif receive_cnt < send_num:
    sys.stdout.flush()
      
def set_environ(): #windows下测试用函数，设置环境变量
  os.environ['mq_account']='reid'
  os.environ['mq_pwd']='reid'
  os.environ['mq_ip']='192.168.101.151'
  os.environ['mq_port']= '5672'

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--num', type=int, default = 100)
  parser.add_argument('--img_path', type=str, default = './car')
  args = parser.parse_args()
  #获取mq配置
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
  # 向队列插入数值 routing_key是队列名
  message = generate_msg(image_path=args.img_path)
  for i in range(args.num):
    channel.basic_publish(exchange = '',routing_key = 'Coll2Reid',body = message)
  print('successfully send %s message to ReID consumer' %args.num)
  print('*******************************')
  start_time = time.time()
  send_num = args.num

  # 告诉rabbitmq，用callback来接收ReID consumer返回消息
  channel.basic_consume('Reid2Coll',callback)
  # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理
  print('Ready to receive result from ReID consumer')
  channel.start_consuming()
  
