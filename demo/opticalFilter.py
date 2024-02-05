import cv2
import sys
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
from tianmoucv.alg import *

class opticalDetector_Maxone():
    
    def __init__(self,noiseThresh=8,distanceThresh=0.2):
        self.noiseThresh = noiseThresh
        self.th = distanceThresh
        self.accumU = 0
        self.accumV = 0
        
    def __call__(self,sd,td,ifInterploted = False):
        
        td[abs(td)<self.noiseThresh] = 0
        sd[abs(sd)<self.noiseThresh] = 0
       
        #rawflow = cal_optical_flow(sd,td,win=7,stride=3,mask=None,ifInterploted = ifInterploted)
        #rawflow = recurrentOF(sd,td,ifInterploted = ifInterploted)
        rawflow = recurrentMultiScaleOF(sd,td,ifInterploted = ifInterploted)
        
        flow = flow_to_image(rawflow.permute(1,2,0).numpy())
        
        flowup = np.zeros([flow.shape[0]*2,flow.shape[1]*2,3])
        flowup[1::2,1::2,:] = flow/255.0
        flowup[0::2,1::2,:] = flow/255.0
        flowup[1::2,0::2,:] = flow/255.0
        flowup[0::2,0::2,:] = flow/255.0

        u = rawflow.permute(1,2,0).numpy()[:, :, 0]
        v = rawflow.permute(1,2,0).numpy()[:, :, 1]
        uv = [u,v]

        distance = ((u)**2 + (v)**2) *(u<0)
        
        distance[distance>self.th] = 1
        distance[distance<self.th] = 0
        distanceup = np.zeros([flow.shape[0]*2,flow.shape[1]*2])

        kernel = np.ones((3,3),np.uint8)              
        distance = cv2.dilate(distance,kernel,iterations=3) 

        distanceup[1::2,1::2] = distance * 255.0
        distanceup[0::2,1::2] = distance * 255.0
        distanceup[1::2,0::2] = distance * 255.0
        distanceup[0::2,0::2] = distance * 255.0
        f = (distanceup).copy().astype(np.uint8)
        contours,hierarchy = cv2.findContours(f,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(f, contours, -1, (0, 255, 255), 2)

        area = []
        box = None
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        if len(area)>0:
            if np.max(area) < 1200:
                return None,distanceup,flowup
            max_idx = np.argmax(area)
            for i in range(max_idx - 1):
                cv2.fillConvexPoly(f, contours[max_idx - 1], 0)
            cv2.fillConvexPoly(f, contours[max_idx], 255)

            maxcon = contours[max_idx]
            x1 = np.min(maxcon[:,:,0])  
            x2 = np.max(maxcon[:,:,0])  
            y1 = np.min(maxcon[:,:,1])  
            y2 = np.max(maxcon[:,:,1])  
            box = [x1,y1,x2,y2]
            #print(u[y1//2:y2//2,x1//2:x2//2]>0)
            #print(u[y1//2:y2//2,x1//2:x2//2],v[y1//2:y2//2,x1//2:x2//2])

        return box,distanceup,flowup