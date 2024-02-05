#0917 version

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


######################################################################################################
##UNET
######################################################################################################
class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, instance_norm=True,relu=False):

        super(ConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None
        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)
        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.instance_norm:
            out = self.instance(out)
        if self.relu:
            out = self.relu(out)
        return out

class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 反卷积 layer 用于上采样
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        # Crop x2 到 x1 的尺寸以进行跳跃链接 skip connection
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)


class UNet(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    def forward(self, x):
        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        x = self.down4(s4)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = torch.clamp(self.conv3(x),0,1)
        return x

    
######################################################################################################
#AttnNet
######################################################################################################
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        Att_score = self.spatial_attention(out)
        out =Att_score * out
        return out,Att_score



class AttnNet(nn.Module):
    
    def __init__(self, inputchannel=7, img_channel = 3):
        # Initialize neural network blocks.
        
        super(AttnNet, self).__init__()
        
        self.img_channel = img_channel
        self.conv1 = nn.Conv2d(inputchannel, 64, 1, stride=1, padding=0)
        self.CBAM1 = CBAM(64)
        self.down1 = down(64, 128, 3)
        self.CBAM2 = CBAM(128)
        self.up1   = up(128, 64)
        self.conv3 = nn.Conv2d(64, img_channel, 1, stride=1, padding=0)
        
    def forward(self, I_0, I0_reco_t):

        ref_img = torch.cat([I_0, I0_reco_t],dim=1)
        s1 = torch.tanh(self.conv1(ref_img))
        s1_att,atts1 = self.CBAM1(s1)
        x = self.down1(s1_att)
        x,atts2 = self.CBAM2(x)
        x = self.up1(x, s1_att)
        ResDiff = self.conv3(x)
        img_ref = ResDiff
       
        return  img_ref, ResDiff


######################################################################################################
#OF
######################################################################################################
class SpyNet(torch.nn.Module):
    def __init__(self,dim=2+2+1):
        super().__init__()

        class Basic(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=dim, out_channels=32, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, stride=1, padding=2)
                )
            def forward(self, tenInput):
                return self.netBasic(tenInput)
            
        self.N_level = 6
        self.input_dim = dim
        self.netBasic = torch.nn.ModuleList([ Basic(dim+2) for intLevel in range(self.N_level) ])
        self.backwarp_tenGrid = {}

    def backwarp(self, tenInput, tenFlow):
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - \
                                    (1.0 / tenFlow.shape[3]),
                                    tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - \
                                    (1.0 / tenFlow.shape[2]), 
                                    tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)

        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), \
                             tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.shape)] \
                                + tenFlow).permute(0, 2, 3, 1), \
                                mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, TD, SD_0, SD_t):
        
        SD_0_list = [SD_0]
        SD_t_list = [SD_t]
        TD_list = [TD]
        
        for intLevel in range(self.N_level-1):
            SD_0_list.append(torch.nn.functional.avg_pool2d(input=SD_0_list[-1], \
                                       kernel_size=2, stride=2, count_include_pad=False))
            SD_t_list.append(torch.nn.functional.avg_pool2d(input=SD_t_list[-1], \
                                       kernel_size=2, stride=2, count_include_pad=False))
            TD_list.append(torch.nn.functional.avg_pool2d(input=TD_list[-1], \
                                       kernel_size=2, stride=2, count_include_pad=False))

        tenFlow = torch.zeros([ SD_0_list[-1].size(0), 2, \
                    int(math.floor(SD_0_list[-1].size(2) / 2.0)), int(math.floor(SD_0_list[-1].size(3) / 2.0)) ])
        
        tenFlow = tenFlow.to(TD.device)
        
        for intLevel in range(self.N_level):
            
            invert_index = self.N_level-intLevel-1
            Flow_upsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, \
                                                           mode='bilinear', align_corners=True) * 2.0
            
            if Flow_upsampled.size(2) != SD_0_list[invert_index].size(2): \
                Flow_upsampled = torch.nn.functional.pad(input=Flow_upsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if Flow_upsampled.size(3) != SD_0_list[invert_index].size(3): \
                Flow_upsampled = torch.nn.functional.pad(input=Flow_upsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')
            
            
            tenFlow = self.netBasic[intLevel](torch.cat([ SD_0_list[invert_index], \
                                                          self.backwarp(SD_t_list[invert_index], Flow_upsampled), \
                                                          TD_list[invert_index],\
                                                          Flow_upsampled ], 1)) + Flow_upsampled
        return tenFlow

    
######################################################################################################
#Merged
######################################################################################################
class TianmoucRecon(nn.Module):

    def __init__(self,imgsize):
        super(TianmoucRecon, self).__init__()
        self.flowComp = SpyNet(dim=1+2+2)
        self.syncComp = UNet(8, 3)
        self.AttnNet = AttnNet(inputchannel=3+1,img_channel=3)
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
        imgOut = torch.nn.functional.grid_sample(img, grid)
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
        
        # Part1. 极致的RGB修复
        I0 = F0
        TFlow_1_0  = -1 * TFlow_0_1 
   
        # Part2. 光流重建，针对小幅度运动
        F_1_0 = self.flowComp(TFlow_1_0, SD0, SD1) #输出值0~1
        I_1_0 = self.backWarp(I0, F_1_0)

        # part3. 最简单的时间重建，针对大幅度运动
        T_1_t,_ = self.AttnNet(I0,TFlow_0_1)
        #T_1_t = I_1_0

        # part4. 进一步融合

        I_t_p = self.syncComp(torch.cat([SD1,T_1_t,I_1_0],dim=1))


        return I_t_p,F_1_0,T_1_t,I_1_0