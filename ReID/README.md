##Requirements


- python >=3.6
- pytorch 1.1.0 (gpu)
- rabbitMQ

##Test

下载模型权重：链接：https://pan.baidu.com/s/1QhZJHfBncMyU7FvkX27daA 
提取码：bsr3 

在producer路径下运行producer.py,发送测试图片到reid consumer，在consumer路径下运行consumer.py, consumer返回匹配结果给producer，然后producer保存结果到本地并可视化


















