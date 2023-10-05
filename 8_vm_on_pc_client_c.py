#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg 
#import math
#import numpy as np
#import traitlets
#------------------------------------

import json
import trt_pose.coco
with open('/home/xigong/trt_pose/tasks/hand_pose_new/preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(hand_pose)
print("topology has been set successfully--nis\n")
#------------------------------------

import trt_pose.models
num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
print("model has been loaded successfully--nis\n")
#------------------------------------

import torch
WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
print("example data has been set successfully--nis\n")
#------------------------------------

import os
if not os.path.exists('/home/xigong/trt_pose/tasks/hand_pose_new/model/resnest/hand_pose_resnet18_att_244_244_trt.pth'):
    print("the model has not been optimized yet--nis\n")
    MODEL_WEIGHTS = '/home/xigong/trt_pose/tasks/hand_pose_new/model/resnest/hand_pose_resnet18_att_244_244.pth'
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    import torch2trt
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    OPTIMIZED_MODEL = '/home/xigong/trt_pose/tasks/hand_pose_new/model/resnest/hand_pose_resnet18_att_244_244_trt.pth'
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    print("the model has been optimized --nis\n")
else:
    print("the optimized model has exsited --nis\n")
    OPTIMIZED_MODEL = '/home/xigong/trt_pose/tasks/hand_pose_new/model/resnest/hand_pose_resnet18_att_244_244_trt.pth'
#------------------------------------

from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print("the optimized model has been loaded --nis\n")
#------------------------------------

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
parse_objects = ParseObjects(topology,cmap_threshold=0.15, link_threshold=0.15)
draw_objects = DrawObjects(topology)
print("visual callable classes has been set --nis\n")
#------------------------------------

import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#nis
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def draw_joints(image, joints):
    count = 0
    for i in joints:
        if i==[0,0]:
            count+=1
    if count>= 3:
        return 
    for i in joints:
        cv2.circle(image, (i[0],i[1]), 2, (0,0,255), 1)
    cv2.circle(image, (joints[0][0],joints[0][1]), 2, (255,0,255), 1)
    for i in hand_pose['skeleton']:
        if joints[i[0]-1][0]==0 or joints[i[1]-1][0] == 0:
            break
        cv2.line(image, (joints[i[0]-1][0],joints[i[0]-1][1]), (joints[i[1]-1][0],joints[i[1]-1][1]), (255,255,255), 1)#draw the joints on image to show

print("image preprocess has been set --nis\n")
#------------------------------------

from preprocess.preprocessdata import preprocessdata
preprocessdata = preprocessdata(topology, num_parts)

import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))

svm_train = False
if svm_train:
    clf, predicted = preprocessdata.trainsvm(clf, joints_train, joints_test, hand.labels_train, hand.labels_test)
    filename = '/home/xigong/trt_pose/tasks/hand_pose/svmmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
else:
    filename = '/home/xigong/trt_pose/tasks/hand_pose_new/model/sklearn/svmmodel_plus.sav'
    clf = pickle.load(open(filename, 'rb'))

with open('/home/xigong/trt_pose/tasks/hand_pose_new/preprocess/gesture.json', 'r') as f:
    gesture = json.load(f)
gesture_type = gesture["classes_nis"]

print("ges_clf preprocess has been set --nis\n")
#------------------------------------
from jetcam.usb_camera import USBCamera

camera = USBCamera(width=WIDTH, height=HEIGHT)#here device need to be adjusted
print("camera has been set --nis\n")
#------------------------------------

from adafruit_servokit import ServoKit
myCameraKit=ServoKit(channels=16)
import time 

ANGLE_PHI = 90
ANGLE_THET = 25

def motor_control(joints_x, joints_y, text):
    global WIDTH
    global HEIGHT
    global ANGLE_PHI  
    global ANGLE_THET  
    if text != "pan":
        if joints_x != 0 and joints_y != 0:
            if joints_x-0.5*WIDTH > 0.3*WIDTH:
                if ANGLE_PHI < 180:
                    ANGLE_PHI += 1
                    myCameraKit.servo[15].angle=ANGLE_PHI
            elif joints_x-0.5*WIDTH < -0.3*WIDTH:       
                if ANGLE_PHI > 0:
                    ANGLE_PHI -= 1
                    myCameraKit.servo[15].angle=ANGLE_PHI
            
            if joints_y-0.5*HEIGHT > 0.2*HEIGHT:#方向颠倒
                if ANGLE_THET > 0:
                    ANGLE_THET -= 1
                    myCameraKit.servo[14].angle=ANGLE_THET
            elif joints_y-0.5*HEIGHT < -0.2*HEIGHT:
                if ANGLE_THET < 90:
                    ANGLE_THET += 1
                    myCameraKit.servo[14].angle=ANGLE_THET
#------------------------------------

def image_show(data_q):
    while True:
        image_s = data_q.get()
        cv2.imshow("USB Camera0", image_s)
        cv2.waitKey(1)
        #time.sleep(0.1)
#------------------------------------

def execute(data_q):
    while True:
        cursor_joint = 0
        
        image = camera.read()#read a frame
        image_s = cv2.flip(image, 1)
        
        data = preprocess(image_s)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        joints = preprocessdata.joints_inference(image_s, counts, objects, peaks)
        draw_joints(image_s, joints)#image process

        dist_bn_joints = preprocessdata.find_distance(joints)
        gesture = clf.predict([dist_bn_joints,[0]*num_parts*num_parts])
        gesture_joints = gesture[0]
        preprocessdata.prev_queue.append(gesture_joints)
        preprocessdata.prev_queue.pop(0)
        preprocessdata.print_label(image_s, preprocessdata.prev_queue, gesture_type)#ges_clf

        data_q.put(image_s)
        joint_x = joints[12][0]
        joint_y = joints[12][1]
        data = [preprocessdata.text, joint_x, joint_y, ANGLE_PHI, ANGLE_THET]  # 这里替换为要发送的实际数据
        serialized_data = pickle.dumps(data)
        try:
            client.sendall(serialized_data)  # 发送数据，需要将字符串编码为字节流
        except BrokenPipeError:
            print("Server disconnected --nis")
            client.close()
            break

        motor_control(joint_x, joint_y, preprocessdata.text)

#------------------------------------ 
import socket
import pickle
import queue
import threading
import time

# 创建一个TCP/IP Socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 定义PC的IP地址和端口号
pc_ip = '192.168.1.99'
pc_port = 12345
# 连接到PC的Socket
client.connect((pc_ip, pc_port))
print("Server connected --nis")

# 循环发送数据
data_q = queue.Queue(maxsize=6)

imshow_thread = threading.Thread(target=image_show, args=(data_q, ))
imshow_thread.daemon = True
execute_thread = threading.Thread(target=execute, args=(data_q, ))

myCameraKit.servo[15].angle=ANGLE_PHI
myCameraKit.servo[14].angle=ANGLE_THET#云台初始化
execute_thread.start()
time.sleep(2)
imshow_thread.start()
myCameraKit.servo[15].angle=ANGLE_PHI
myCameraKit.servo[14].angle=ANGLE_THET#云台归位