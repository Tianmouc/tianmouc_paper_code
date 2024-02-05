import numpy as np
from scipy.signal import convolve2d
import cv2


##Author: Taoyi Wang
def lyncam_raw_comp(raw):
    width = raw.shape[1] * 2
    height = raw.shape[0] * 1
    #raw_res = np.zeros((height, width), dtype=np.int16)
    #raw_res = cv2.resize(raw, (width, height),interpolation=cv2.INTER_LINEAR)
    raw_hollow = np.zeros((height, width), dtype=np.int16)
    test =   raw[0::4, 3::2]
    raw_hollow[0::4, 2::4], raw_hollow[2::4, 0::4] = raw[0::4, 0::2], raw[2::4, 0::2]
    raw_hollow[0::4, 3::4], raw_hollow[2::4, 1::4] = raw[0::4, 1::2], raw[2::4, 1::2]
    raw_hollow[1::4, 2::4], raw_hollow[3::4, 0::4] = raw[1::4, 0::2], raw[3::4, 0::2]
    raw_hollow[1::4, 3::4], raw_hollow[3::4, 1::4] = raw[1::4, 1::2], raw[3::4, 1::2]
    comp_kernal = np.array([[0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0]], dtype=np.float32) * 0.25
    raw_comp = np.zeros_like(raw_hollow)
    cv2.filter2D(raw_hollow, -1, comp_kernal, raw_comp, anchor= (-1, -1), borderType=cv2.BORDER_ISOLATED)
    raw_comp = raw_comp + raw_hollow;
    raw_comp = raw_comp.astype(np.uint16)
    return raw_comp


##Author: Taoyi Wang
def demosaicing_npy(bayer=None,  level=0 ,bitdepth=8):
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
    green = np.zeros_like(bayer_cal)
    blue = np.zeros_like(bayer_cal)

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
   
    rgb = np.stack([red, green, blue], -1).clip(min=0, max=max_v).astype(dtype)
    return rgb

