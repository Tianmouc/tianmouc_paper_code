import argparse
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(BASE_DIR)
import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter
import random
import sys
import libs.dataset as dataset
from libs.utils import DataLoaderX, plot_img_and_mask,plot_one_box,show_seg_result
from libs.config import cfg
from libs.config import update_config
from libs.core.loss import get_loss
from libs.core.function import validate
from libs.core.general import fitness
from libs.models import get_net
from libs.utils.utils import create_logger, select_device
from libs.core.general import check_img_size,  non_max_suppression,scale_coords
import cv2

def show_seg_result(img, result, index, epoch,palette=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 128, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[result == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (640,320), interpolation=cv2.INTER_LINEAR) 
    return img,color_mask

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')
    args = parser.parse_args(args=[])
    args.modelDir = ""
    args.weights = "./"
    args.logDir = "runs/"
    args.modelDir = "./YOLOPv1/weights"
    args.conf_thres = 0.25
    args.iou_thres = 0.5
    return args

class YOLOPv1detector():
    
    def __init__(self, path, raw_size = (320, 640), input_size = (320,640), device=None):
        
        device = torch.device('cuda:0') if device is None else device
        
        args = parse_args()
        update_config(cfg, args)
        self.device = device
        model = get_net(cfg)
        print("finish build model")
        model_dict = model.state_dict()
        checkpoint_file = path
        checkpoint = torch.load(checkpoint_file)
        checkpoint_dict = checkpoint['state_dict']
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)

        self.model = model.to(device)
        self.model.gr = 1.0
        self.model.nc = 10
        print('bulid model finished')

        max_stride = 32
        _, imgsz = [check_img_size(x, s=max_stride) for x in cfg.MODEL.IMAGE_SIZE] #imgsz is multiple of max_stride

        test_batch_size = 1
        nc = 10
        
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        self.model.eval()
        self.cfg = cfg
        self.shapes = ((320, 640), ((1, 1), (0, 0)))

        self.names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
                
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def _get_normalizer(self,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.normalize = transforms.Normalize(mean=mean, std=std)
        
    def __call__(self,img_raw,conf_thresh=0.5,iou_thres=0.6):
        epoch = 0
        img = self.normalize(img_raw)
        img_np = img_raw.cpu()[0,...].permute(1,2,0).numpy()
        
        nb, _, height, width = img.size()    #batch size, channel, height, widt
        with torch.no_grad():
            pad_w, pad_h = self.shapes[1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = self.shapes[1][0][0]

            det_out, da_seg_out, ll_seg_out= self.model(img)
            inf_out,train_out = det_out
            
            _,da_predict=torch.max(da_seg_out, 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            _,ll_predict=torch.max(ll_seg_out, 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            output = non_max_suppression(inf_out, conf_thres=conf_thresh, iou_thres=iou_thres)

            img_test  = img_np.copy()
            da_seg_mask = da_seg_out[0][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
            da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
            img_seg_safe,safe_mask = show_seg_result(img_test, da_seg_mask, 0,epoch,None)

            img_ll  = img_np.copy()
            ll_seg_mask = ll_seg_out[0][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
            ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
            img_ll1 = img_ll.copy()
            img_seg_line,line_mask = show_seg_result(img_ll, ll_seg_mask, 0,epoch,None)

            img_det  = img_np.copy()
            det = output[0].clone()
            if len(det):
                det[:,:4] = scale_coords(img[0].shape[1:],det[:,:4],img_det.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img_det , label=label_det_pred, color=[0,0,255], line_thickness=2)

            return img_det,img_seg_safe,img_seg_line,det,inf_out