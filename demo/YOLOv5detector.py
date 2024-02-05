import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(BASE_DIR)
sys.path.append('./YOLOv5')
import torchvision.transforms as transforms
import argparse
import time
from pathlib import Path
import cv2
import torch

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box


def get_opt():
    parser = argparse.ArgumentParser(description='Test Multitask network')
    opt = parser.parse_args(args=[])
    opt.iou_thres= 0.5
    opt.conf_thres = 0.25
    opt.imgsz= 320
    opt.task= 'val'
    opt.device= torch.device("cuda:0")
    opt.half= False
    opt.data = './YOLOv5/data/bdd100k.yaml' # check YAML
    opt.cfg = './YOLOv5/models/yolov5_tianmouc.yaml '
    print_args(vars(opt))
    return opt

opt = get_opt()


class YOLOv5detector_SD():
    
    def __init__(self,path):
        
        device= opt.device
        iou_thres= opt.iou_thres
        dnn = False
        half = opt.half 
        data = opt.data
    
        # Load model
        self.model = DetectMultiBackend(path, device=device, dnn=dnn, data=data, fp16=half)
        self.device  = opt.device
        stride, pt, jit, engine = self.model.stride, self.model.pt, self.model.jit, self.model.engine
        half = self.model.fp16  # FP16 supported on limited backends with CUDA
        self.model.eval()

        
    def __call__(self,SD,img_raw):
        
        self.model.warmup(imgsz=SD.shape)  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
 
        t1 = time_sync()
        im0 = img_raw

        #SD = SD.half() if self.model.fp16 else im.float()  # uint8 to fp16/32

        t2 = time_sync()
        dt[0] += t2 - t1

        #print('yolov5sdinfo:',SD.shape,torch.max(SD),'~',torch.min(SD))
        out = self.model(SD, augment=False, visualize=False)

        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(out, opt.conf_thres, opt.iou_thres, classes=None, max_det=100)

        dt[2] += time_sync() - t3
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        return pred,out
               


