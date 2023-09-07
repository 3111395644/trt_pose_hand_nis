import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import math
import numpy as np
import traitlets
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

from jetcam.usb_camera import USBCamera
#from jetcam.csi_camera import CSICamera
#from jetcam.utils import bgr8_to_jpeg

#camera = CSICamera(width=WIDTH, height=HEIGHT)
camera = USBCamera(width=WIDTH, height=HEIGHT)#here device need to be adjusted
print("camera has been set --nis\n")
#------------------------------------

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

from preprocessdata import preprocessdata
preprocessdata = preprocessdata(topology, num_parts)
#from gesture_classifier import gesture_classifier
#gesture_classifier = gesture_classifier()

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
        cv2.line(image, (joints[i[0]-1][0],joints[i[0]-1][1]), (joints[i[1]-1][0],joints[i[1]-1][1]), (0,255,0), 1)

print("preprocess has been set --nis\n")
#------------------------------------

import time

frame_count = 0
start_time = time.time()

while(True):#nis
    image = camera.read()#read a frame
    
    elapsed_time = time.time() - start_time#display FPS
    fps = frame_count / elapsed_time
    cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    frame_count += 1

    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    joints = preprocessdata.joints_inference(image, counts, objects, peaks)
    draw_joints(image, joints)
    
    cv2.imshow("USB Camera0", image)
    kk = cv2.waitKey(1)
    if kk == ord('q'):  # press q then quit
        print("camera out --nis\n")
        break



 
    



