#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg 
#import math
#import numpy as np
#import traitlets
#------------------------------------

import json
import trt_pose.coco
with open('/home/xigong/trt_pose/tasks/hand_pose/preprocess/hand_pose.json', 'r') as f:
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
if not os.path.exists('/home/xigong/trt_pose/tasks/hand_pose/model/hand_pose_resnet18_att_244_244_trt.pth'):
    print("the model has not been optimized yet--nis\n")
    MODEL_WEIGHTS = '/home/xigong/trt_pose/tasks/hand_pose/model/hand_pose_resnet18_att_244_244.pth'
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    import torch2trt
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    OPTIMIZED_MODEL = '/home/xigong/trt_pose/tasks/hand_pose/model/hand_pose_resnet18_att_244_244_trt.pth'
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    print("the model has been optimized --nis\n")
else:
    print("the optimized model has exsited --nis\n")
    OPTIMIZED_MODEL = '/home/xigong/trt_pose/tasks/hand_pose/model/hand_pose_resnet18_att_244_244_trt.pth'
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

from preprocessdata import preprocessdata
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
    filename = '/home/xigong/trt_pose/tasks/hand_pose/svmmodel_plus.sav'
    clf = pickle.load(open(filename, 'rb'))

with open('/home/xigong/trt_pose/tasks/hand_pose/preprocess/gesture.json', 'r') as f:
    gesture = json.load(f)
gesture_type = gesture["classes_nis"]

print("ges_clf preprocess has been set --nis\n")
#------------------------------------

from jetcam.usb_camera import USBCamera

camera = USBCamera(width=WIDTH, height=HEIGHT)#here device need to be adjusted
print("camera has been set --nis\n")
#------------------------------------
import pyautogui
import time

pyautogui.FAILSAFE = False

def control_cursor(data_q):
    ScreenWidth, ScreenHeight = pyautogui.size()
    Cur_x, Cur_y = 0,0
    Pre_x, Pre_y = 0,0
    Delay_L, Delay_R = 0,0
    Delay_U, Delay_D = 0,0
    MouseSensitivity, ScrollSensitivity = 224, 1#for adjust
    
    while(True):
        text = data_q.get()[0]
        joints_pos_x = data_q.get()[1]
        joints_pos_y = data_q.get()[2]
        Cur_x = (joints_pos_x)*ScreenWidth/MouseSensitivity
        Cur_y = (joints_pos_y)*ScreenHeight/MouseSensitivity
            
        if text == "pan":
            Delay_L, Delay_R = 0,0
            Delay_U, Delay_D = 0,0
            if abs(Cur_x-Pre_x)<102 and abs(Cur_y-Pre_y)<53:#hpf
                pyautogui.move(-(Cur_x-Pre_x), Cur_y-Pre_y)
        elif text == "fist":
            Delay_L += 1
            if Delay_L > 5:
                pyautogui.click(duration=0.3)
                Delay_L = 0
        elif text == "palm":
            Delay_R += 1
            if Delay_R > 5:
                pyautogui.click(button="right",duration=0.3)
                Delay_R = 0
        elif text == "thumb_up":
            Delay_U += 1
            if Delay_U > 5:
                pyautogui.scroll(1)
                Delay_U = 0
        elif text == "ok":
            Delay_D += 1
            if Delay_D > 5:
                pyautogui.scroll(-1)
                Delay_D = 0
        
        Pre_x = Cur_x
        Pre_y = Cur_y
                    
print("cursor control has been set --nis\n")

#------------------------------------
def execute(data_q):
    cursor_joint = 8

    while(True):#nis
        image = camera.read()#read a frame
        
        data = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        joints = preprocessdata.joints_inference(image, counts, objects, peaks)
        draw_joints(image, joints)#image process

        dist_bn_joints = preprocessdata.find_distance(joints)
        gesture = clf.predict([dist_bn_joints,[0]*num_parts*num_parts])
        gesture_joints = gesture[0]
        preprocessdata.prev_queue.append(gesture_joints)
        preprocessdata.prev_queue.pop(0)
        image_s = cv2.flip(image, 1)
        preprocessdata.print_label(image_s, preprocessdata.prev_queue, gesture_type)#ges_clf
        
        data_q.put([preprocessdata.text, joints[cursor_joint][0], joints[cursor_joint][1]])

        cv2.imshow("USB Camera0", image_s)
        quitkey = cv2.waitKey(1)
        if quitkey == ord('q'):  # press q then quit
            print("camera out --nis\n")
            break

#------------------------------------  
import queue
import threading

data_q = queue.Queue(maxsize=6)

control_cursor_thread = threading.Thread(target=control_cursor, args=(data_q, ))
control_cursor_thread.daemon = True
execute_thread = threading.Thread(target=execute, args=(data_q, ))

execute_thread.start()
time.sleep(1)
control_cursor_thread.start()

execute_thread.join()
