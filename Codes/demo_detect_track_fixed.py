from __future__ import division

import sys

import queue
import pdb
import os
import time
import datetime
import argparse
import struct
import socket
import json
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from collections import deque
import numpy as np
from detect_wrapper.Detectoruav import DroneDetection
from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker

# sys.path.append((os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__),'detect_wrapper'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper','dronetracker'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper','drtracker'))




## Input video - Fixed path for Linux
video_path = os.path.join(os.path.dirname(__file__),'testvideo','testvideo','n16.mp4')

magnification = 2

import warnings
warnings.filterwarnings("ignore")

# IP, Port = 0, 0
# IP = "192.168.0.139"
# Port = 9874
# AF_INET, SOCK_DGRAM = socket.AF_INET, socket.SOCK_DGRAM
# udp_socket = socket.socket(AF_INET, SOCK_DGRAM)
# udp_socket.bind(("",9921))

def lefttop2center(bbx):
    obbx=[0,0,bbx[2],bbx[3]]
    obbx[0]=bbx[0]+bbx[2]/2
    obbx[1]=bbx[1]+bbx[3]/2
    return obbx

# class  DroneTracker():
#     def __init__(self):
#         self.tracker = drtracker.CorrFilterTracker(False, True, True)

#     def init_track(self, init_box, init_frame):
#         self.tracker.init(init_box, init_frame)
#     def on_track(self, frame):
#         boundingbox = self.tracker.update(frame)
#         boundingbox = [int(x) for x in boundingbox]
#         return boundingbox

def test():
    # Fixed path - use os.path.join instead of hardcoded backslashes
    IRweights_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'detect_wrapper','weights','best.pt')

    # Verify files exist
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        print(f"Available videos:")
        testvideo_dir = os.path.join(os.path.dirname(__file__),'testvideo','testvideo')
        if os.path.exists(testvideo_dir):
            for f in os.listdir(testvideo_dir):
                print(f"  - {f}")
        return

    if not os.path.exists(IRweights_path):
        print(f"ERROR: Weights not found at {IRweights_path}")
        return

    print(f"Loading video from: {video_path}")
    print(f"Loading weights from: {IRweights_path}")

    time_record=[]
    det_time=[]
    interval=50
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return

    # vw= VideoWriter("/home/dell/Project_UAV/result/result.avi", fps=30)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame from video")
        return

    print(f"Video frame shape: {frame.shape}")
    # pdb.set_trace()
    oframe = frame.copy()
    drone_det=DroneDetection(IRweights_path=IRweights_path, RGBweights_path=IRweights_path)
    drone_tracker =Tracker()  #DroneTracker()
    first_track=True
    frame_count = 0
    while(ret):
        frame_count += 1
        t1=time.time()

        init_box=drone_det.forward_IR(frame)
        t2=time.time()
        det_time.append(t2-t1)
        # if vw is not None:
        #     vw.write(oframe)
        if init_box is not None:
            if first_track:
                drone_tracker.init_track(init_box,frame)
                # first_track=False
            else:
                drone_tracker.change_state(init_box)
            bbx = [int(x) for x in init_box]
            print(f"Frame {frame_count}: Detection box {bbx}")

            # cv2.rectangle(oframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
            # cv2.imshow("tracking", oframe)
            # cv2.waitKey(1)

            visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
            bbx=[int(i*magnification) for i in bbx]
            cv2.rectangle(visuframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
            cv2.imshow("tracking", visuframe)
            cv2.waitKey(1)

        num=0
        while(num<interval):
            num=num+1
            ret, frame = cap.read()


            if ret:
                frame_count += 1
                oframe = frame.copy()
                visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
                t1=time.time()
                outputs=drone_tracker.on_track(frame)
                t2=time.time()
                time_record.append(t2-t1)
                bbx=[int(i*magnification) for i in outputs]
                print(f"Frame {frame_count}: Tracking box {bbx}")

                cv2.rectangle(visuframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
                cv2.imshow("tracking", visuframe)
                cv2.waitKey(1)
                # if vw is not None:
                #     vw.write(oframe)
    cap.release()
    # if vw is not None:
    #     vw.release()
    cv2.destroyAllWindows()
    print("done......")
    print('track average time:',np.array(time_record).mean())
    print('detect average time:',np.array(det_time).mean())

    # Calculate and display FPS
    if len(time_record) > 0:
        tracking_fps = 1.0 / np.array(time_record).mean()
        print(f'tracking FPS: {tracking_fps:.2f}')
    if len(det_time) > 0:
        detection_fps = 1.0 / np.array(det_time).mean()
        print(f'detection FPS: {detection_fps:.2f}')
    if len(time_record) > 0 and len(det_time) > 0:
        total_avg_time = (np.array(time_record).mean() + np.array(det_time).mean())
        overall_fps = 1.0 / total_avg_time if total_avg_time > 0 else 0
        print(f'overall FPS (detection + tracking): {overall_fps:.2f}')

if __name__=="__main__":
    test()
