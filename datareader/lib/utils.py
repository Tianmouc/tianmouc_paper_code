import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.dirname(__file__))
import struct

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


def load_lyncam_bin(img_fpath, rod_or_cone, rod_adc_bit, img_per_file):
    
    if rod_adc_bit == 8:
        one_frm_size = 0x9e00
    elif rod_adc_bit == 4:
        one_frm_size = 0x4D00
    elif rod_adc_bit == 2:
        one_frm_size = 0x1C40
    else:
        print("ADC precision Not support")
        return None
    
    size = os.path.getsize(img_fpath)
    size_int = size // 4
    frm_head_offset = 16
    assert size_int // one_frm_size == img_per_file
    
    #if rod_or_cone == "cone":
        
    if rod_or_cone == "cone":
        raw = np.zeros(size_int - frm_head_offset, dtype=np.int16)
        with open(img_fpath, 'rb') as f:
            for i in range(size_int):
                data = f.read(4) # read 4byte for int32
                realdata = struct.unpack('i',data) # unpack to 32bit int
                data_int = realdata[0]
                #if data_int == -1:
                #   print("{}, {}, int {}, hex {}".format(i, data, data_int, hex(data_int)))
                #pvalue_np[i] = data_int
                if i >= frm_head_offset:
                    raw[i - frm_head_offset] = data_int & 0x3ff        
        return raw
    elif rod_or_cone == "rod":
        #pvalue_np = np.zeros((img_per_file, one_frm_size), dtype=np.int32)
        pvalue = []
        with open(img_fpath, 'rb') as f:
            for img in range(img_per_file):
                data = f.read(one_frm_size)
                pvalue.append(data)
                #pvalue_np[0] = 1
                #for i in range()
        return pvalue
    else:
        return None


def visualize_tdiff(tdiff_out, tdiff_vis_gain, invert=True):
    '''
    if self.use_torch:
        tdiff_out = tdiff_out.cpu() if self.torch_dev == 'cuda' else tdiff_out
        tdiff_out = tdiff_out.numpy()
    '''
    height, width = tdiff_out.shape
    tdiff_out = cv2.resize(tdiff_out.astype(np.float32), (width*2, height))
    height, width = tdiff_out.shape
    tdiff_vis = np.zeros((height, width, 3), dtype=np.uint8)
    tdiff = (tdiff_out * tdiff_vis_gain).clip(min=-255, max=255)
    tdiff_pos = tdiff * (tdiff > 0)
    tdiff_neg = -tdiff * (tdiff < 0)

    tdiff_vis[..., 0] = tdiff_pos.astype(np.uint8)
    tdiff_vis[..., 1] = tdiff_neg.astype(np.uint8)
    if invert:
        white = np.ones_like(tdiff_vis) * 255
        tdiff_vis = white - tdiff_vis
    return tdiff_vis

def visualize_sdiff(sdiff_l, sdiff_r, sdiff_vis_gain, invert=True, raw = False):

        height, width  = sdiff_l.shape
        #sdiff_vis_gain = 1.5
        sdiff_vis = np.zeros((height, width * 2, 3), dtype=np.uint8)
        sdiff = np.zeros((height, width * 2), dtype=np.int64)
        sdiff[:, 0::2] = sdiff_l
        sdiff[:, 1::2] = sdiff_r

        sdiffv = (sdiff * sdiff_vis_gain).clip(min=-255, max=255)
        sdiff_pos = sdiffv * (sdiffv > 0)
        sdiff_neg = -sdiffv * (sdiffv < 0)

        sdiff_pos [sdiff_pos < 0] = 0
        sdiff_neg [sdiff_neg < 0] = 0
        # sdiff_neg = sdiff_neg.astype(np.uint8)

        sdiff_vis[..., 0] = sdiff_pos.astype(np.uint8)
        sdiff_vis[..., 1] = sdiff_neg.astype(np.uint8)

        if invert:
            white = np.ones_like(sdiff_vis) * 255
            sdiff_vis = white - sdiff_vis
      
        if raw:
            return sdiff_vis, sdiff
        else:
            return sdiff_vis

def rm_r_ts_offset(r_ts, ts_init):
    r_now = (r_ts - ts_init) * 10 / 1000
    return r_now
