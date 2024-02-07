from re import L
import numpy as np
import cv2
from scipy.signal import convolve2d
from math import log
# import lib.utils as utils
import sys
import os
sys.path.append(os.path.dirname(__file__))
from lib import utils
#from utils import get_lp_mask
import torch
from torch.nn import functional as F
def rgb2gray(rgb, type='ISBN', device='cpu'):
    if device == 'cpu':
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        if type == "CCIR_601":
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        elif type == 'ISBN':  # same as opencv
            gray = 0.256999969 * r + 0.50399971 * g + 0.09799957 * b
    elif device == 'torch':
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        if type == "CCIR_601":
            gray = torch.mul(0.299,  r) + torch.mul(0.587, g) + torch.mul(0.114, b)
        elif type == 'ISBN':  # same as opencv
            gray = torch.mul(0.256999969, r) + torch.mul(0.50399971, g) + torch.mul(0.09799957, b)
    return gray


def rgb2bayer(rgb=None):
    assert rgb.shape[2] == 3
    bayer = np.zeros((rgb.shape[0], rgb.shape[1]))
    bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]  # red
    bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]  # blue
    bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]  # green red
    bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]  # green blue
    return bayer

def demosaicing_torch(bayer, bayer_pattern, bitdepth=8):
        # 1st step: reconstruct rgb values at all pixels using algorithm from Henrique Malvar et.al., ICASSP, 2004.
        # simple bilinear interpolation first
        #max_v = torch.max(bayer.cpu())
        max_v = (2 ** bitdepth) - 1
        conv_p, conv_c, conv_ud, conv_lr = torch.zeros([1,1,3,3]), torch.zeros([1,1,3,3]), torch.zeros([1,1,3,3]), torch.zeros([1,1,3,3])
        red, green, blue = torch.zeros_like(bayer), torch.zeros_like(bayer), torch.zeros_like(bayer)
        
        conv_p[0,0,:,:] = torch.FloatTensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) * 1 / 4  # plus shaped interpolation
        conv_c[0,0,:,:] = torch.FloatTensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) * 1 / 4  # cross shaped interpolation
        conv_ud[0,0,:,:] = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 1, 0]]) * 1 / 2  # up and down interpolation        
        conv_lr[0,0,:,:] = torch.FloatTensor([[0, 0, 0], [1, 0, 1], [0, 0, 0]]) * 1 / 2  # left and right interpolation
        p = F.conv2d(bayer.unsqueeze(0).unsqueeze(0), conv_p, stride = 1, padding = 1).squeeze(0).squeeze(0)
        c = F.conv2d(bayer.unsqueeze(0).unsqueeze(0), conv_c, stride = 1, padding = 1).squeeze(0).squeeze(0)
        ud = F.conv2d(bayer.unsqueeze(0).unsqueeze(0), conv_ud, stride = 1, padding = 1).squeeze(0).squeeze(0)
        lr = F.conv2d(bayer.unsqueeze(0).unsqueeze(0), conv_lr, stride = 1, padding = 1).squeeze(0).squeeze(0)
           
        # center gradient v1 for plus shaped interpolation
        conv_grad_c1 = torch.zeros([1,1,5,5])
        conv_grad_c1[0,0,:,:] =  torch.FloatTensor([[0, 0, -1, 0, 0],
                                                    [0, 0, 0, 0, 0],
                                                    [-1, 0, 4, 0, -1],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, -1, 0, 0]]) * 1 / 8
        conv_grad_c2 = torch.zeros([1,1,5,5])               
        # center gradient v2 for cross shaped interpolation
        conv_grad_c2[0,0,:,:] =  torch.FloatTensor([[0, 0, -3 / 2, 0, 0],
                                                    [0, 0, 0, 0, 0],
                                                    [-3 / 2, 0, 6, 0, -3 / 2],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, -3 / 2, 0, 0]]) * 1 / 8
        conv_grad_v = torch.zeros([1,1,5,5])                               
        # horizontal gradient for left and right interpolation
        conv_grad_v[0,0,:,:] = torch.FloatTensor([[0, 0, -1, 0, 0],
                                                   [0, -1, 0, -1, 0],
                                                   [1/2, 0, 5, 0, 1/2],
                                                   [0, -1, 0, -1, 0],
                                                   [0, 0, -1, 0, 0]]) * 1 / 8
        conv_grad_h = torch.zeros([1,1,5,5])                           
        # horizontal gradient for left and right interpolation
        conv_grad_h[0,0,:,:] =  torch.FloatTensor([[0, 0, 1 / 2, 0, 0],
                                                   [0, -1, 0, -1, 0],
                                                   [-1, 0, 5, 0, -1],
                                                   [0, -1, 0, -1, 0],
                                                   [0, 0, 1 / 2, 0, 0]]) * 1 / 8
        # calculate gradient for compensation
        grad_c1 = F.conv2d(bayer.unsqueeze(0).unsqueeze(0),  conv_grad_c1, stride = 1, padding = 2).squeeze(0).squeeze(0)
        grad_c2 = F.conv2d(bayer.unsqueeze(0).unsqueeze(0),  conv_grad_c2, stride = 1, padding = 2).squeeze(0).squeeze(0)
        grad_h = F.conv2d(bayer.unsqueeze(0).unsqueeze(0), conv_grad_h, stride = 1, padding = 2).squeeze(0).squeeze(0)
        grad_v = F.conv2d(bayer.unsqueeze(0).unsqueeze(0), conv_grad_v, stride = 1, padding = 2).squeeze(0).squeeze(0)
       
        if bayer_pattern == 'rggb':
            red[0::2, 0::2] = bayer[0::2, 0::2]
            red[0::2, 1::2] = lr[0::2, 1::2]
            red[1::2, 0::2] = ud[1::2, 0::2]
            red[1::2, 1::2] = c[1::2, 1::2]

            green[0::2, 0::2] = p[0::2, 0::2]
            green[0::2, 1::2] = bayer[0::2, 1::2]
            green[1::2, 0::2] = bayer[1::2, 0::2]
            green[1::2, 1::2] = p[1::2, 1::2]

            blue[0::2, 0::2] = c[0::2, 0::2]
            blue[0::2, 1::2] = ud[0::2, 1::2]
            blue[1::2, 0::2] = lr[1::2, 0::2]
            blue[1::2, 1::2] = bayer[1::2, 1::2]
            # add gradient compensation
            red[0::2, 1::2] += grad_h[0::2, 1::2]
            red[1::2, 0::2] += grad_v[1::2, 0::2]
            red[1::2, 1::2] += grad_c2[1::2, 1::2]

            green[0::2, 0::2] += grad_c1[0::2, 0::2]
            green[1::2, 1::2] += grad_c1[1::2, 1::2]

            blue[0::2, 0::2] += grad_c2[0::2, 0::2]
            blue[0::2, 1::2] += grad_v[0::2, 1::2]
            blue[1::2, 0::2] += grad_h[1::2, 0::2]
        elif bayer_pattern == 'grbg':
            red[0::2, 0::2] = lr[0::2, 0::2]
            red[0::2, 1::2] = bayer[0::2, 1::2]
            red[1::2, 0::2] = c[1::2, 0::2]
            red[1::2, 1::2] = ud[1::2, 1::2]
            
            green[0::2, 0::2] = bayer[0::2, 0::2]
            green[0::2, 1::2] = p[0::2, 1::2]
            green[1::2, 0::2] = p[1::2, 0::2]
            green[1::2, 1::2] = bayer[1::2, 1::2]
            
            blue[0::2, 0::2] = ud[0::2, 0::2]
            blue[0::2, 1::2] = c[0::2, 1::2]
            blue[1::2, 0::2] = bayer[1::2, 0::2]   
            blue[1::2, 1::2] = lr[1::2, 1::2]
            # add gradient compensation
            red[0::2, 0::2] += grad_h[0::2, 0::2]
            red[1::2, 1::2] += grad_v[1::2, 1::2]
            red[1::2, 0::2] += grad_c2[1::2, 0::2]
            
            green[0::2, 1::2] += grad_c1[0::2, 1::2]
            green[1::2, 0::2] += grad_c1[1::2, 0::2]
            
            blue[0::2, 0::2] += grad_v[0::2, 0::2]
            blue[0::2, 1::2] += grad_c2[0::2, 1::2]
            blue[1::2, 1::2] += grad_h[1::2, 1::2]
        elif bayer_pattern == 'bggr':
            blue[0::2, 0::2] = bayer[0::2, 0::2]
            blue[0::2, 1::2] = lr[0::2, 1::2]
            blue[1::2, 0::2] = ud[1::2, 0::2]
            blue[1::2, 1::2] = c[1::2, 1::2]

            green[0::2, 0::2] = p[0::2, 0::2]
            green[0::2, 1::2] = bayer[0::2, 1::2]
            green[1::2, 0::2] = bayer[1::2, 0::2]
            green[1::2, 1::2] = p[1::2, 1::2]

            red[0::2, 0::2] = c[0::2, 0::2]
            red[0::2, 1::2] = ud[0::2, 1::2]
            red[1::2, 0::2] = lr[1::2, 0::2]
            red[1::2, 1::2] = bayer[1::2, 1::2]
            # add gradient compensation
            blue[0::2, 1::2] += grad_h[0::2, 1::2]
            blue[1::2, 0::2] += grad_v[1::2, 0::2]
            blue[1::2, 1::2] += grad_c2[1::2, 1::2]

            green[0::2, 0::2] += grad_c1[0::2, 0::2]
            green[1::2, 1::2] += grad_c1[1::2, 1::2]

            red[0::2, 0::2] += grad_c2[0::2, 0::2]
            red[0::2, 1::2] += grad_v[0::2, 1::2]
            red[1::2, 0::2] += grad_h[1::2, 0::2]
        #print("r max",red.max(), "g max",green.max(),"b max",blue.max())
        colored = torch.cat([red.unsqueeze(-1), green.unsqueeze(-1), blue.unsqueeze(-1)], dim = -1).clamp(0, max_v)
        return colored

def RAW_WB_torch(bayer, bayer_pattern, fixed_gain, bitdepth=8):
    max_v = (2 ** bitdepth) - 1
    r_gain = fixed_gain[0]
    gr_gain = fixed_gain[1]
    gb_gain = fixed_gain[2]
    b_gain = fixed_gain[3]
    bayer_awb = torch.zeros_like(bayer)
    if bayer_pattern == 'rggb':
        bayer_awb[0::2, 0::2] = bayer[0::2, 0::2] * r_gain
        bayer_awb[0::2, 1::2] = bayer[0::2, 1::2] * gr_gain
        bayer_awb[1::2, 0::2] = bayer[1::2, 0::2] * gb_gain
        bayer_awb[1::2, 1::2] = bayer[1::2, 1::2] * b_gain
    elif bayer_pattern == 'grbg':
        bayer_awb[0::2, 0::2] = bayer[0::2, 0::2] * gr_gain
        bayer_awb[0::2, 1::2] = bayer[0::2, 1::2] * r_gain
        bayer_awb[1::2, 0::2] = bayer[1::2, 0::2] * b_gain
        bayer_awb[1::2, 1::2] = bayer[1::2, 1::2] * gb_gain
    elif bayer_pattern == 'bggr':
        bayer_awb[0::2, 0::2] = bayer[0::2, 0::2] * b_gain
        bayer_awb[0::2, 1::2] = bayer[0::2, 1::2] * gb_gain
        bayer_awb[1::2, 0::2] = bayer[1::2, 0::2] * gr_gain
        bayer_awb[1::2, 1::2] = bayer[1::2, 1::2] * r_gain
    bayer_awb = bayer_awb.clamp(0, max_v)
    return bayer_awb

def RGB_GAMMA_torch(rgb, gamma):
    max_v = 255.0
    rgb_gamma = max_v * pow(((rgb) / max_v), gamma)
    return rgb_gamma

def RAW_BLC_torch(bayer, bayer_pattern, bl_params, bitdepth=8):
    max_v = (2 ** bitdepth) - 1
    r_bl = bl_params[0]
    gr_bl = bl_params[1]
    gb_bl = bl_params[2]
    b_bl = bl_params[3]
    alpha = bl_params[4]
    beta = bl_params[5]
    bayer_blc = torch.zeros_like(bayer)
    if bayer_pattern == 'rggb':
        bayer_blc[0::2, 0::2] = bayer[0::2, 0::2] * r_bl
        bayer_blc[0::2, 1::2] = bayer[0::2, 1::2] * gr_bl + alpha * bayer[0::2, 0::2] / 256
        bayer_blc[1::2, 0::2] = bayer[1::2, 0::2] * gb_bl + beta * bayer[1::2, 1::2] / 256
        bayer_blc[1::2, 1::2] = bayer[1::2, 1::2] * b_bl
    elif bayer_pattern == 'grbg':
        bayer_blc[0::2, 0::2] = bayer[0::2, 0::2] * gr_bl + alpha * bayer[0::2, 1::2] / 256
        bayer_blc[0::2, 1::2] = bayer[0::2, 1::2] * r_bl 
        bayer_blc[1::2, 0::2] = bayer[1::2, 0::2] * b_bl
        bayer_blc[1::2, 1::2] = bayer[1::2, 1::2] * gb_bl + beta * bayer[1::2, 0::2] / 256
    elif bayer_pattern == 'bggr':
        bayer_blc[0::2, 0::2] = bayer[0::2, 0::2] * b_bl
        bayer_blc[0::2, 1::2] = bayer[0::2, 1::2] * gb_bl + beta * bayer[0::2, 0::2] / 256
        bayer_blc[1::2, 0::2] = bayer[1::2, 0::2] * gr_bl + alpha * bayer[1::2, 1::2] / 256
        bayer_blc[1::2, 1::2] = bayer[1::2, 1::2] * r_bl
    bayer_blc = bayer_blc.clamp(0, max_v)
    return bayer_blc    
    

def bayer2rgb(bayer=None, bayer_pattern='rggb', level=1, torch_or_np='np', raw_bit_depth=8):
    if torch_or_np == 'np':
        rgb = demosaicing_npy(bayer=bayer, bayer_pattern=bayer_pattern, level=level)
    elif torch_or_np == 'torch':
        #bayer_pattern = 'rggb' if bayer_type == 0 else \
        #                'grbg' if bayer_type == 1 else \
        #                'bggr' if bayer_type == 2 else \
        #                'gbrg'if bayer_type == 3 else \
        #                None
        #bitdepth = 8 if bayer.dtype=='uint8' else 12 # if bayer.dtype=='uint16'
        if type(bayer) is np.ndarray:
            bayer = torch.tensor(bayer, dtype=torch.float)
            #bayer = torch.from_numpy(bayer)
        #print("bayer max ",bayer.max())
        rgb = demosaicing_torch(bayer=bayer, bayer_pattern=bayer_pattern, bitdepth = raw_bit_depth)
    return rgb


# Compute psnr between 2 image
def psnr(img1, img2):
    mse = np.std(img1 - img2)
    return 20 * np.log10(255 / mse)


# Compute psnr between 2 RGB image
def psnr_rgb(img1, img2):
    assert img1.shape[2] == img2.shape[2] == 3
    img1_fp = img1.astype(np.float32)
    img2_fp = img2.astype(np.float32)
    mse_r = np.std(img1_fp[..., 0] - img2_fp[..., 0])  # Red
    mse_b = np.std(img1_fp[..., 2] - img2_fp[..., 2])  # Blue
    mse_g = np.std(img1_fp[..., 1] - img2_fp[..., 1])  # Green
    mse_ave = (mse_g + mse_b + mse_r) / 3.0
    return 20 * np.log10(255 / mse_ave)

# implemented using numpy, decrepted

def demosaicing_npy(bayer=None, bayer_pattern='rggb', level=0 ,bitdepth=8):
    """
    Call this function to load raw bayer image
    :param bayer: input bayer image
    :param level: demosaicing level. 0: bilinear linear; 1: gradient
    :param bayer_type: bayer_type: : 0--RGrRGr...GbBGbB, 1--GrRGrR...BGbBGb...
  
    """
    assert bayer is not None
    dtype = bayer.dtype
    max_v = (2 ** bitdepth) - 1#bayer.max()
    bayer_cal = bayer.astype(np.float32)
    # 1st step: standard bilinear demosaicing process on bayer-patterned image
    conv_p = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) * 1 / 4  # plus shaped interpolation
    conv_c = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float32) * 1 / 4  # cross shaped interpolation
    conv_ud = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32) * 1 / 2  # up and down interpolation
    conv_lr = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.float32) * 1 / 2  # left and right interpolation
    p = convolve2d(bayer_cal, conv_p, boundary='symm', mode='same')
    c = convolve2d(bayer_cal, conv_c, boundary='symm', mode='same')
    ud = convolve2d(bayer_cal, conv_ud, boundary='symm', mode='same')
    lr = convolve2d(bayer_cal, conv_lr, boundary='symm', mode='same')
    # calculate gradient
    # center gradient v1 for plus shaped interpolation
    conv_grad_c1 = np.array([[0,    0,  -1, 0,  0],
                                [0,    0,  0,  0,  0],
                                [-1,   0,  4,  0,  -1],
                                [0,    0,  0,  0,  0],
                                [0,    0,  -1, 0,  0]]) * 1 / 8
    # center gradient v2 for cross shaped interpolation
    conv_grad_c2 = np.array([[0,    0,  -3/2,   0,  0],
                                [0,    0,  0,      0,  0],
                                [-3/2, 0,  6,      0,  -3/2],
                                [0,    0,  0,      0,  0],
                                [0,    0, -3/2,    0,  0]]) * 1 / 8
    # horizontal gradient for left and right interpolation
    conv_grad_h = np.array([[0, 0, 1 / 2, 0, 0],
                            [0, -1, 0, -1, 0],
                            [-1, 0, 5, 0, -1],
                            [0, -1, 0, -1, 0],
                            [0, 0, 1 / 2, 0, 0]]) * 1 / 8
    # vertical gradient for up and down interpolation
    conv_grad_v = conv_grad_h.T
    grad_c1 = convolve2d(bayer_cal, conv_grad_c1, boundary='symm', mode='same')
    grad_c2 = convolve2d(bayer_cal, conv_grad_c2, boundary='symm', mode='same')
    grad_h = convolve2d(bayer_cal, conv_grad_h, boundary='symm', mode='same')
    grad_v = convolve2d(bayer_cal, conv_grad_v, boundary='symm', mode='same')

    red = np.zeros_like(bayer_cal)
    ''' red[0::2, 0::2] = bayer_cal[0::2, 0::2]
    red[0::2, 1::2] = lr[0::2, 1::2]
    red[1::2, 0::2] = ud[1::2, 0::2]
    red[1::2, 1::2] = c[1::2, 1::2]'''

    green = np.zeros_like(bayer_cal)
    '''green[0::2, 0::2] = p[0::2, 0::2]
    green[0::2, 1::2] = bayer_cal[0::2, 1::2]
    green[1::2, 0::2] = bayer_cal[1::2, 0::2]
    green[1::2, 1::2] = p[1::2, 1::2]'''

    blue = np.zeros_like(bayer_cal)
    '''blue[0::2, 0::2] = c[0::2, 0::2]
    blue[0::2, 1::2] = ud[0::2, 1::2]
    blue[1::2, 0::2] = lr[1::2, 0::2]
    blue[1::2, 1::2] = bayer_cal[1::2, 1::2]'''

    if bayer_pattern == 'rggb':
        red[0::2, 0::2] = bayer[0::2, 0::2]
        red[0::2, 1::2] = lr[0::2, 1::2]
        red[1::2, 0::2] = ud[1::2, 0::2]
        red[1::2, 1::2] = c[1::2, 1::2]

        green[0::2, 0::2] = p[0::2, 0::2]
        green[0::2, 1::2] = bayer[0::2, 1::2]
        green[1::2, 0::2] = bayer[1::2, 0::2]
        green[1::2, 1::2] = p[1::2, 1::2]

        blue[0::2, 0::2] = c[0::2, 0::2]
        blue[0::2, 1::2] = ud[0::2, 1::2]
        blue[1::2, 0::2] = lr[1::2, 0::2]
        blue[1::2, 1::2] = bayer[1::2, 1::2]
        # add gradient compensation
        red[0::2, 1::2] += grad_h[0::2, 1::2]
        red[1::2, 0::2] += grad_v[1::2, 0::2]
        red[1::2, 1::2] += grad_c2[1::2, 1::2]

        green[0::2, 0::2] += grad_c1[0::2, 0::2]
        green[1::2, 1::2] += grad_c1[1::2, 1::2]

        blue[0::2, 0::2] += grad_c2[0::2, 0::2]
        blue[0::2, 1::2] += grad_v[0::2, 1::2]
        blue[1::2, 0::2] += grad_h[1::2, 0::2]
    elif bayer_pattern == 'grbg':
        red[0::2, 0::2] = lr[0::2, 0::2]
        red[0::2, 1::2] = bayer[0::2, 1::2]
        red[1::2, 0::2] = c[1::2, 0::2]
        red[1::2, 1::2] = ud[1::2, 1::2]
        
        green[0::2, 0::2] = bayer[0::2, 0::2]
        green[0::2, 1::2] = p[0::2, 1::2]
        green[1::2, 0::2] = p[1::2, 0::2]
        green[1::2, 1::2] = bayer[1::2, 1::2]
        
        blue[0::2, 0::2] = ud[0::2, 0::2]
        blue[0::2, 1::2] = c[0::2, 1::2]
        blue[1::2, 0::2] = bayer[1::2, 0::2]   
        blue[1::2, 1::2] = lr[1::2, 1::2]
        # add gradient compensation
        red[0::2, 0::2] += grad_h[0::2, 0::2]
        red[1::2, 1::2] += grad_v[1::2, 1::2]
        red[1::2, 0::2] += grad_c2[1::2, 0::2]
        
        green[0::2, 1::2] += grad_c1[0::2, 1::2]
        green[1::2, 0::2] += grad_c1[1::2, 0::2]
        
        blue[0::2, 0::2] += grad_v[0::2, 0::2]
        blue[0::2, 1::2] += grad_c2[0::2, 1::2]
        blue[1::2, 1::2] += grad_h[1::2, 1::2]
    elif bayer_pattern == 'bggr':
        blue[0::2, 0::2] = bayer[0::2, 0::2]
        blue[0::2, 1::2] = lr[0::2, 1::2]
        blue[1::2, 0::2] = ud[1::2, 0::2]
        blue[1::2, 1::2] = c[1::2, 1::2]

        green[0::2, 0::2] = p[0::2, 0::2]
        green[0::2, 1::2] = bayer[0::2, 1::2]
        green[1::2, 0::2] = bayer[1::2, 0::2]
        green[1::2, 1::2] = p[1::2, 1::2]

        red[0::2, 0::2] = c[0::2, 0::2]
        red[0::2, 1::2] = ud[0::2, 1::2]
        red[1::2, 0::2] = lr[1::2, 0::2]
        red[1::2, 1::2] = bayer[1::2, 1::2]
        # add gradient compensation
        blue[0::2, 1::2] += grad_h[0::2, 1::2]
        blue[1::2, 0::2] += grad_v[1::2, 0::2]
        blue[1::2, 1::2] += grad_c2[1::2, 1::2]

        green[0::2, 0::2] += grad_c1[0::2, 0::2]
        green[1::2, 1::2] += grad_c1[1::2, 1::2]

        red[0::2, 0::2] += grad_c2[0::2, 0::2]
        red[0::2, 1::2] += grad_v[0::2, 1::2]
        red[1::2, 0::2] += grad_h[1::2, 0::2]
    rgb = np.stack([red, green, blue], -1).clip(min=0, max=max_v).astype(dtype)
    return rgb
'''        
    if level == 0:
        rgb = np.stack([red, green, blue], -1).astype(dtype)
        return rgb
    # lveel 1: reconstruct rgb values at all pixels using algorithm from Henrique Malvar et.al., ICASSP, 2004.
    elif level == 1:
        # calculate gradient
        # center gradient v1 for plus shaped interpolation
        conv_grad_c1 = np.array([[0,    0,  -1, 0,  0],
                                 [0,    0,  0,  0,  0],
                                 [-1,   0,  4,  0,  -1],
                                 [0,    0,  0,  0,  0],
                                 [0,    0,  -1, 0,  0]]) * 1 / 8
        # center gradient v2 for cross shaped interpolation
        conv_grad_c2 = np.array([[0,    0,  -3/2,   0,  0],
                                 [0,    0,  0,      0,  0],
                                 [-3/2, 0,  6,      0,  -3/2],
                                 [0,    0,  0,      0,  0],
                                 [0,    0, -3/2,    0,  0]]) * 1 / 8
        # horizontal gradient for left and right interpolation
        conv_grad_h = np.array([[0, 0, 1 / 2, 0, 0],
                                [0, -1, 0, -1, 0],
                                [-1, 0, 5, 0, -1],
                                [0, -1, 0, -1, 0],
                                [0, 0, 1 / 2, 0, 0]]) * 1 / 8
        # vertical gradient for up and down interpolation
        conv_grad_v = conv_grad_h.T
        grad_c1 = convolve2d(bayer_cal, conv_grad_c1, boundary='symm', mode='same')
        grad_c2 = convolve2d(bayer_cal, conv_grad_c2, boundary='symm', mode='same')
        grad_h = convolve2d(bayer_cal, conv_grad_h, boundary='symm', mode='same')
        grad_v = convolve2d(bayer_cal, conv_grad_v, boundary='symm', mode='same')
        # add gradient compensation
        red[0::2, 1::2] += grad_h[0::2, 1::2]
        red[1::2, 0::2] += grad_v[1::2, 0::2]
        red[1::2, 1::2] += grad_c2[1::2, 1::2]

        green[0::2, 0::2] += grad_c1[0::2, 0::2]
        green[1::2, 1::2] += grad_c1[1::2, 1::2]

        blue[0::2, 0::2] += grad_c2[0::2, 0::2]
        blue[0::2, 1::2] += grad_v[0::2, 1::2]
        blue[1::2, 0::2] += grad_h[1::2, 0::2]'''
        
    
    


def AWBeasy(rgb):
    rgb = rgb.astype(np.float32)
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r *= kr
    g *= kg
    b *= kb
    rgb_awb = np.zeros_like(rgb)
    rgb_awb[:,:,0] = r
    rgb_awb[:,:,1] = g
    rgb_awb[:,:,2] = b
    rgb_awb = rgb_awb.astype(np.uint16)
    return rgb_awb
