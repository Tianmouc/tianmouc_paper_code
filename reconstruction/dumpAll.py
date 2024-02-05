#python dumpAll.py --ckptver '0710'
import time, os, random,sys,argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
sys.path.append("./")
sys.path.append("../")
sys.path.append("../datareader")
from tianmoucv.alg import cal_optical_flow,backWarp,flow_to_image,white_balance
from tianmoucv.basic import vizDiff
from tianmoucv.nn import warp_fast,interpolate_preprocess
from tianmoucv.isp import lyncam_raw_comp,demosaicing_npy
from tianmoucData import TianmoucDataReader

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--ckptver',type = str,default='0710')
arg = parser.parse_args()

def images_to_video(frame_list,name,Val_size=(512,256),Flip=False):
    fps = 30          
    size = (Val_size[0]*2, Val_size[1]*2) 
    out = cv2.VideoWriter(name, 0x7634706d, fps, size)
    for frame in frame_list:
        frame = (frame[[2,1,0],:,]*255).cpu().permute(1,2,0).numpy() 
        w = Val_size[0]
        h = Val_size[1]
        frame[0:h,0:w,...] = white_balance(frame[0:h,0:w,...])
        frame[h:2*h,w:2*w,...] = white_balance(frame[h:2*h,w:2*w,...])
        if Flip:
            frame[0:h,0:w,:] = frame[h:0:-1,0:w,:]
            frame[h:2*h,0:w,:] = frame[2*h:h-1:-1,0:w,:]
            frame[0:h,w:2*w,:] = frame[h:0:-1,w:2*w,:]
            frame[h:2*h,w:2*w,:] = frame[h*2:h-1:-1,w:2*w,:]
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()

dataset_top1 = "../../data/tianmouc_raw_data"
datasetList = [dataset_top1]

##################### Step1. Env Preparation #####################
local_rank = 0
device = torch.device('cuda:'+str(local_rank))
torch.cuda.set_device(local_rank)
writer = None 
master = False 
CHECKPOINT_DIR =  './ckpts/'
###################### Step2. model and data Preparation #############

from  model.reconstructor_new import TianmoucRecon

CHECKPOINT_PATH_MODEL = './ckpts/best.ckpt'

VALIDATION_BATCH_SIZE = 1
TRAINING_CONTINUE = True
h = 320
w = 640 
Val_size   = (w,h)
ReconModel = TianmoucRecon(Val_size)
ReconModel.load_model(ckpt=CHECKPOINT_PATH_MODEL)
ReconModel.to(device)
start = time.time()

endID = 40
for datasetName in datasetList:
    dirlist = os.listdir(datasetName)
    for key in dirlist:
        dataset = TianmoucDataReader([datasetName],matchkey=key)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=VALIDATION_BATCH_SIZE,\
                                             num_workers=4, pin_memory=False, drop_last = False)
        imlist = []
        with torch.no_grad():
            validationIndex = 0
            startTime  = time.time()
            for index,sampleRaw in enumerate(dataloader, 0):
                if index > endID:
                    break
            
                sample = dict([])
                F0 = sampleRaw['F0']
                F1 = sampleRaw['F1']

                tsdiff = sampleRaw['tsdiff']
                sample['F0'] = F0.permute(0,3,1,2).to(device)
                sample['F1'] =  F1.permute(0,3,1,2).to(device)
                sample['tsdiff'] = tsdiff.to(device)

                middleTime  = time.time()
                F1t, F0,tsdiff= warp_fast(sample,ReconModel,None,h,w,device,ifsingleDirection=True)
                
                tsdiff = tsdiff.cpu()
                for t in range(25):
                    retImg1 = F0.cpu()[t,:,:,:]
                    retImg2 = F1t.cpu()[t,:,:,:]
                    imageCanve = torch.zeros([3,w*4,h*3])
                    gapw = w//4
                    gaoh = h//4
                    imageCanve[:,gaoh:gaoh+h,gapw:gapw+w]

                    sd  = tsdiff[0,1,t,...] * 255      
                    rgb_sd = vizDiff(sd,thresh=12)

                    td = tsdiff[0,0,t,...] * 255    
                    rgb_td = vizDiff(td,thresh=12)

                    img_col1 = torch.cat([retImg1,rgb_td],dim=1)
                    img_col2 = torch.cat([rgb_sd,retImg2],dim=1) 
                    img = torch.cat([img_col1,img_col2],dim=2)
                    imlist.append(img)

                validationIndex += 1
        endTime  = time.time()
        print(key,'/',datasetName, ' cost:',endTime-startTime,'s', ' speed:',(endID*25)/(endTime-startTime),'fps')
        if not os.path.exists('../../results/ver'+arg.ckptver):
            os.mkdir('../../results/ver'+arg.ckptver) 
        vdname = '../../results/ver'+arg.ckptver+'/'+arg.ckptver+'ver_'+key+'@'+datasetName.split('/')[-2]+'.mp4'
        print(vdname)
        images_to_video(imlist,vdname,Val_size=(w,h),Flip=False)
