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
def cv2_sobel(img_in, weight=[0.5, 0.5], X=True, Y=True, gamma=2.2, blur=False):
    img = cv2.GaussianBlur(img_in, (3,3), 0) if blur is True else img_in
    X_AND_Y = X and Y
    if X is True:
        grad_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
    if Y is True:
        grad_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)
        grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(grad_x, weight[0], grad_y, weight[1], 2.2) if X_AND_Y is True else grad_x if X is True else grad_y if Y is True else None
    sobel = cv2.convertScaleAbs(sobel) if sobel is not None else None
    return sobel


def cv2_laplacian(img_in, ksize=3, blur=True):
    img = cv2.GaussianBlur(img_in, (3,3), 0) if blur is True else img_in
    laplace = cv2.Laplacian(img_in, cv2.CV_64F, ksize=ksize)
    laplace = cv2.convertScaleAbs(laplace)
    laplace = cv2.convertScaleAbs(laplace) if laplace is not None else None
    return laplace

def gamma_correct(gray, gamma=2.2, perceptual=False, percp_val=100):
    #gray_gamma = 100 * np.sqrt(pow((gray / 255.0), gamma))
    gray_gamma = 255.0 * pow(((gray) / 255.0), gamma) if perceptual is False else percp_val * np.sqrt(pow((gray / 255.0), gamma))
    return gray_gamma

def HOG_GRAY_MAG(gray, bitdepth=8, mode='linear'):
    #mode = liner, or log
    kx = np.array([ [0, 0, 0],
                    [-1, 0, 1],
                    [ 0, 0, 0]])
    ky = np.array([ [0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]])
    #kx = np.array([-1, 0, 1])
    gray[(gray - 0) < 1e-6]  +=  1.0
    img_in = gray if mode == 'linear' else np.log10(gray) if mode == 'log' else None
    gx = cv2.filter2D(img_in,-1,kx)
    gy = cv2.filter2D(img_in,-1,ky)
    #gx2 = gx * gx
    #gy2 = gy * gy
    grad = np.sqrt(gx * gx + gy * gy)
    if mode == 'linear':
        if bitdepth == 8:
            grad = grad / 16
            img_conv = grad.astype(np.uint8)
        elif bitdepth == 3:
            grad = grad / 512
            img_conv = grad.astype(np.uint8)
            img_conv = img_conv * 36
        img_conv = 255 - img_conv
    elif mode == 'log':
        if bitdepth == 2:
            dt=2
            img_conv = np.zeros(grad.shape, dtype=np.uint8)          
            for x in range(0, grad.shape[1]):
                for y in range(0, gray.shape[0]):
                    #pix = 10 ** grad[y, x]
                    pix = grad[y, x]
                    if pix > dt:
                        img_conv[y,x] = 0
                    elif pix < 1/dt:
                        img_conv[y,x] = 255
                    else:
                        img_conv[y,x] = 128
        if bitdepth == 3:
            dt1, dt2, dt3 = 1.3, 2.0, 4.2
            img_conv = np.zeros(grad.shape, dtype=np.uint8)          
            for x in range(0, grad.shape[1]):
                for y in range(0, gray.shape[0]):
                    #pix = 10 ** grad[y, x]
                    pix = grad[y, x]
                    if pix > dt3:
                        img_conv[y,x] = 0
                    elif pix < dt3 and pix > dt2:
                        img_conv[y,x] = 43
                    elif pix < dt2 and pix > dt1:
                        img_conv[y,x] = 86
                    elif pix < dt1 and pix > 1/dt1:
                        img_conv[y,x] = 126 #255
                    elif pix < 1/dt1 and pix > 1/dt2:
                        img_conv[y,x] = 169 #212
                    elif pix < 1/dt2 and pix > 1/dt3:
                        img_conv[y,x] = 212#169
                    elif pix < 1/dt3:
                        img_conv[y,x] = 255#126
                    
            
    return img_conv

def HOG_GRAY_MAG_log(gray, bitdepth=3):
    maxv = np.log10(4095/1)
    gx = np.zeros(gray.shape, dtype=np.float64)
    gy = np.zeros(gray.shape, dtype=np.float64)
    gray_pad = np.pad(gray, 1, mode='edge')
    gray_pad[(gray_pad - 0) < 1e-6]  +=  1.0
    for x in range(0, gray.shape[1]):
        for y in range(0, gray.shape[0]):
            xl, xr = gray_pad[y+1, x], gray_pad[y+1, x+2]
            yu, yd = gray_pad[y, x+1], gray_pad[y+2, x+1]
            gx[y, x] = xr / xl
            gy[y, x] = yd / yu
    gx = np.log10(gx)
    gy = np.log10(gy)
    grad = np.sqrt(gx * gx + gy * gy) / maxv
    if bitdepth == 8:
        #grad = grad / 16
        grad = grad * 255
        img_conv = grad.astype(np.uint8)
    elif bitdepth == 3:
        grad = grad * 8
        img_conv = grad.astype(np.uint8) * 32
    img_conv = 255 - img_conv
    return img_conv
            
    

def gray_ctrst_enhance(gray, gamma=2.2):
    #  Bi, Y., & Andreopoulos, Y. (2017, September). "PIX2NVS" 2017 IEEE ICIP
    lum = 100 * np.sqrt(pow((gray / 255.0), gamma))
    gray_ctrst_enhance = np.zeros_like(lum)
    lum = np.pad(lum, ((1,1), (1,1)), 'edge')
    for y in range(0, gray_ctrst_enhance.shape[0]):
        for x in range(0, gray_ctrst_enhance.shape[1]):
            y_l = y + 1
            x_l = x + 1
            gray_ctrst_enhance[y, x] = (abs(lum[y_l,x_l] - lum[y_l,x_l-1]) + abs(lum[y_l,x_l] - lum[y_l, x_l+1]) + abs(lum[y_l, x_l] - lum[y_l-1, x_l]) + abs(lum[y_l, x_l] - lum[y_l+1, x_l])) / 4.0
    return gray_ctrst_enhance

def log_intesnsity(gray, th = 20):
    #  Bi, Y., & Andreopoulos, Y. (2017, September). "PIX2NVS" 2017 IEEE ICIP
    l = np.zeros_like(gray)
    for y in range(0, gray.shape[0]):
        for x in range(0, gray.shape[1]):
            l[y, x] = log(gray[y,x]) if gray[y,x] > th else gray[y,x]
    return l


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

def bayer_interpolation(bayer, sdiff, gc_coeff):
        #max_v = torch.max(bayer.cpu())
        grad_conv_w = torch.zeros([1,1,3,3])
        grad_conv_w[0,0,:,:] = torch.FloatTensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) * 1 / 4
        conv_w = torch.zeros([1,1,5,5])
        conv_w[0,0,0, 2] = conv_w[0,0,2,0] = conv_w[0,0,2,4] =conv_w[0,0,4,2] = 1 / 4
        # 1st step: reconstruct spacial gradient at odd row large pixels
        grad = torch.mean(sdiff,dim=-1)
        
        grad_recon = F.conv2d(grad.unsqueeze(0).unsqueeze(0), grad_conv_w, stride = 1, padding = 1).squeeze(0).squeeze(0)

        grad[0::2, 0::2] = grad_recon[0::2, 0::2]

        # grad is at half resolution compared with bayer, upsample by simple copying
        grad_up = torch.zeros_like(bayer)

        r_coff = 0.299
        gr_coff = 0.587
        gb_coff = 0.587
        b_coff = 0.114

        grad_up[0::2, 0::2] = grad * r_coff  # red grad
        grad_up[0::2, 1::2] = grad * gr_coff
        grad_up[1::2, 0::2] = grad * gb_coff
        grad_up[1::2, 1::2] = grad * b_coff
        # 2nd step: fill large pixels with bayer values, using gradient corrected bilinear interpolation

        lp_rgb = F.conv2d(bayer.unsqueeze(0).unsqueeze(0), conv_w, stride = 1, padding = 2).squeeze(0).squeeze(0)
        lp_mask = torch.FloatTensor(utils.get_lp_mask(*bayer.size()))
        bayer_intp = bayer + lp_rgb * lp_mask + grad_up * lp_mask * gc_coeff
        return bayer_intp
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

def rgb2hsv_torch(rgb=None, bit_depth=8):
    
    return

def rgb2yuv_torch(rgb=None, bit_depth=8):
    
    return
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
        
    
    


