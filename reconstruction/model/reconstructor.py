#0704 version
from .network import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TianmoucRecon(nn.Module):

    def __init__(self,imgsize):
        super(TianmoucRecon, self).__init__()
        self.flowComp = SpyNet(dim=1+2+2)
        self.syncComp = UNet(8, 3)
        self.AttnNet = UNetRecon(4, 3)
        self.W, self.H = imgsize
        self.gridX, self.gridY = np.meshgrid(np.arange(self.W), np.arange(self.H))

    def backWarp(self, img, flow):
        # Extract horizontal and vertical flows.
        MAGIC_num = 0.5
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        gridX = torch.tensor(self.gridX, requires_grad=False, device=flow.device)
        gridY = torch.tensor(self.gridY, requires_grad=False, device=flow.device)
        x = gridX.unsqueeze(0).expand_as(u).float() + u + MAGIC_num
        y = gridY.unsqueeze(0).expand_as(v).float() + v + MAGIC_num
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid,align_corners=False)
        return imgOut     
    
    def load_model(self,ckpt=None):
        dict1 = torch.load(ckpt, map_location=torch.device('cpu'))
        self.load_state_dict(dict1['state_dict_ReconModel'],strict=True)
        
    def save_model(self,epoch,lr,valLoss,valPSNR,ckpt):
        dict_this_param = {
                        'Detail':"My define.",'epoch':epoch,
                        'learningRate':lr,
                        'valLoss':valLoss,
                        'valPSNR':valPSNR,
                        'state_dict_ReconModel':self.state_dict(),
                        }
        torch.save(dict_this_param, ckpt)   
          
    
    def forward(self, F0, TFlow_0_1, SD0, SD1):
        TFlow_1_0  = -1 * TFlow_0_1 
        # Part2. warp
        F_1_0 = self.flowComp(TFlow_1_0, SD0, SD1) #
        I_1_warp = self.backWarp(F0, F_1_0)

        # part3. time integration
        I_1_rec = self.AttnNet(torch.cat([F0,TFlow_0_1],dim=1))#3+1

        # part4. fusion
        I_t_p = self.syncComp(torch.cat([I_1_rec,I_1_warp,SD1],dim=1))#3+3+2

        return I_t_p,F_1_0,I_1_rec,I_1_warp