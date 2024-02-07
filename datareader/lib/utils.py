import numpy as np
import cv2
import os
from time import time
# import imageio
import pandas as pd
from tqdm import tqdm
# from scipy.signal import convolve2d
from PIL import Image
import sys
sys.path.append(os.path.dirname(__file__))
#import lib.basic_isp as basic_isp
#from . import basic_isp
from lib import basic_isp
import json
import struct
def get_lp_mask(height, width):
    lp_mask = np.zeros((height, width), dtype=np.int64)
    lp_mask[0::4, 0::4] = lp_mask[0::4, 1::4] = lp_mask[1::4, 0::4] = lp_mask[1::4, 1::4] = 1
    lp_mask[2::4, 2::4] = lp_mask[2::4, 3::4] = lp_mask[3::4, 2::4] = lp_mask[3::4, 3::4] = 1
    return lp_mask


def load_from_file(video_pth):
    video = cv2.VideoCapture(video_pth)

    while True:
        ret, img = video.read()
        if ret is False:
            break

        img = img[..., ::-1]  # flip from BGR to RGB
        yield img

def load_image_csv(img_name, width: int, height: int, target_width, target_height, crp_x_shft, crp_y_shft, crop=True):
    cols = set(range(0, height))
    img = np.loadtxt(img_name, delimiter=",", usecols=cols, dtype=np.float32)#.reshape(height, width)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    if crop == True:
        #img = crop_to_realBayer(img, width, height, target_width, target_height,crp_x_shft=crp_x_shft, crp_y_shft=crp_y_shft)
        x_start = int((width - target_width) / 2)  + crp_x_shft
        y_start = int((height - target_height) / 2) + crp_y_shft
        #cropped = np.zeros([height, width], dtype=rawfile.dtype)
        img = img[y_start:y_start+target_height, x_start:x_start+target_width]
    return img

def load_from_folder(folder_pth, dataset, start=0, length=None, filetype='rgb', sensor_height=320, sensor_width=640, crp_x_shft=0, crp_y_shft=0, do_crop=True, camera = 'mini_ux', exp_interval = 30):
    '''
    Call this function to load dataset from folder

    :param folder_pth: dataset folder path
    :param dataset: dataset name
    :param start: frane start
    :param length: stop = start + length
    :param filetype: raw--raw bayer from camera, rgb: self demosaicing or rendering RGB np array,
    :param sensor_height: sensor height
    :param sensor_width: sensor width
    :return:None
    '''

        #raw_filelist = raw_filelist[:length]
    # raw image: first convert to RGB
    if camera == 'huateng':
        param_file = os.path.join(folder_pth, 'config.json')
        with open(param_file, 'r') as f:
            dataset_param = json.load(f)
        color_fps = dataset_param['color']['fps']
        mono_fps = dataset_param['mono']['fps']
        fps_ratio = mono_fps // color_fps
        assert dataset_param['mono']['capture_number'] // dataset_param['color']['capture_number'] == fps_ratio
        rgb_raw_root_path = os.path.join(folder_pth, 'color')
        mono_raw_root_path = os.path.join(folder_pth, 'mono')
        color_flist = sorted(os.listdir(rgb_raw_root_path))
        mono_flist = sorted(os.listdir(mono_raw_root_path))
        interval = fps_ratio
        c_idx = 0
        frame_idx = 0
        for cf in color_flist:
            if filetype == 'raw':
                rgb = np.load(os.path.join(rgb_raw_root_path, cf))
                rgb = rgb[:,:,[2,1,0]]
            elif filetype == 'rgb':
                rgb = cv2.imread(os.path.join(rgb_raw_root_path, cf))
                rgb = rgb[:, :, [2, 1, 0]]
            lp_only = frame_idx % fps_ratio != 0
            if do_crop:
                rgb = crop_RGB(rgb, dataset_param['color']['width'], dataset_param['color']['height'], sensor_width, sensor_height, crp_x_shft=crp_x_shft, crp_y_shft=crp_y_shft)
           # img = rgb
            yield rgb
            for mf in mono_flist[c_idx*interval:(c_idx+1)*interval]:
                if filetype == 'raw':
                    mono = np.load(os.path.join(mono_raw_root_path, mf))
                elif filetype == 'rgb':
                    mono = cv2.imread(os.path.join(mono_raw_root_path, mf))
                    mono = mono[:, :, 0]
                if do_crop:
                    mono = crop_to_realBayer(mono, dataset_param['mono']['width'], dataset_param['mono']['height'], sensor_width, sensor_height, crp_x_shft=crp_x_shft, crp_y_shft=crp_y_shft)
                frame_idx += 1
                yield mono
            c_idx += 1
    elif camera == 'anyverse':
        param_file = os.path.join(folder_pth, 'config.json')
        with open(param_file, 'r') as f:
            dataset_param = json.load(f)        
        width = dataset_param['width']
        height = dataset_param['height']
        cam_name = dataset_param['camera']
        total_captures = dataset_param['total_captures']
        assert start + length <= total_captures
        #assert exp_interval == dataset_param['interval']
        for i in range(start, start + length):
            voltage_fpath =  os.path.join(os.path.join(os.path.join(folder_pth,  str(i)), cam_name), 'voltage_images')
            mono = load_image_csv(os.path.join(voltage_fpath, "denoised_R.csv"), width = width, height = height, target_width = sensor_width, target_height = sensor_height, crp_x_shft = crp_x_shft, crp_y_shft = crp_y_shft, crop=do_crop)
            #yieldif i % exp_interval == 0: # has bayer
            bayer = load_image_csv(os.path.join(voltage_fpath, "denoised_C.csv"), width = width, height = height, target_width = sensor_width, target_height = sensor_height, crp_x_shft = crp_x_shft, crp_y_shft = crp_y_shft, crop=do_crop)

                #yield np.stack([bayer, mono], axis = 0)
            yield (bayer, mono)
            #else: # only mono
            #    yield mono
        
    else:
        file_list = sorted(os.listdir(folder_pth))
        fstart = start if start < len(file_list) else len(file_list) - length
        file_list = file_list[fstart:]
        raw_filelist = []
        
        if length is not None and length < len(file_list):
            file_list = file_list[:length]        
        if filetype == 'raw':
            for i in file_list:
                path = os.path.join(folder_pth, i)
                if os.path.isfile(path):
                    if os.path.splitext(path)[1] == '.raw' or os.path.splitext(path)[1] == '.raww' or os.path.splitext(path)[1] == '.RAW':
                        raw_filelist.append(i)
                #raw_filelist = raw_filelist[fstart:]
                #if length is not None and length < len(raw_filelist):
                #   raw_filelist = raw_filelist[:length]
            if camera == 'mini_ux':
                param_file = os.path.join(folder_pth, dataset) + '.cih'
                dataset_param = get_camera_param(param_file, camera = camera)
            #elif camera == 'huateng':
            #    param_file = os.path.join(folder_pth, 'config.json')
            #    dataset_param = get_camera_param(param_file, camera = camera)
            for file in raw_filelist:
                img = load_raw_img(os.path.join(folder_pth, file), dataset_param,
                                width=sensor_width, height=sensor_height, crp_x_shft=crp_x_shft, crp_y_shft=crp_y_shft, crop = do_crop, camera = camera)
            #img = basic_isp.demosaicing(bayer, level=1)
                yield img
                #return img
        elif filetype == 'rgb':
            for file in file_list:
                img = np.load(os.path.join(folder_pth, file))
                yield img
        
        else:
            for file in file_list:
                img = cv2.imread(os.path.join(folder_pth, file))[..., ::-1]  # flip from BGR to RGB
                yield img
            #return img

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

def crop_resize(img, height, width):
    img_h, img_w = img.shape[:2]
    if img_h / img_w > height / width:
        cropped_h = int(img_w * height / width)
        margin = (img_h - cropped_h) // 2
        cropped = img[margin:margin + cropped_h]
    else:
        cropped_w = int(img_h * width / height)
        margin = (img_w - cropped_w) // 2
        cropped = img[:, margin:margin + cropped_w]

    resized = cv2.resize(cropped, (width, height))
    return resized


def above_thres(x: np.array, thres):
    return (x - thres) * (x > thres) + (x + thres) * (x < -thres)


# load raw RGB image recorded from Camera
def load_rawRGB_img(filename, dataset_param, bitdepth=8, width=640, height=320):

    assert (bitdepth == 8) or (bitdepth == 16)
    if bitdepth == 8:
        rawfile = np.fromfile(filename, dtype=np.uint8)
    elif bitdepth == 16: #16bit per pixel, real depth is 12bit, 'h0XXX, where X is effective pixel value
        rawfile = np.fromfile(filename, dtype=np.int16)
    org_width = int(dataset_param['width'])
    org_height = int(dataset_param['height'])
    rawfile_rgb = rawfile.reshape(org_height, org_width, 3) #three channel RGB
    rawfile_rgb = crop_RGB(rawfile_rgb, org_width, org_height, width, height)
    return rawfile_rgb


# load one raw image (Bayer CFA format) recorded by camera
def load_raw_img(filename, dataset_param, width=640, height=320, bayer_type=0, out_type = 'bayer', crp_x_shft=0, crp_y_shft=0, crop=True, camera='mini_ux'):
    """
    Call this function to load raw bayer image
    :param filename: raw image to be loaded
    :param dataset_param: dataset parameters
    :param width: Lyncam real width
    :param height: Lyncam real height
    :param bayer_type: 0--RGrRGr...GbBGbB, 1--GrRGrR...BGbBGb...;
    :param out_type: 'bayer', 1channel bayer array(grayscale); 'rgb_3ch', RGB 3 channel Bayer array
    """
    if camera == 'mini_ux':
        bitdepth = int(dataset_param['bitdepth']) # bit depth of the sensor raw data(8, 16...)
        assert (bitdepth == 8) or (bitdepth == 16) or (bitdepth == 12)
        if bitdepth == 8:
            rawfile = np.fromfile(filename, dtype=np.uint8)  # 8bit raw image
        elif bitdepth <= 16 and bitdepth > 8:  # 16bit per pixel, real depth: 12~16bit; if 12bit, 'h0XXX, X is effective pixel value
            rawfile = np.fromfile(filename, dtype=np.uint16)  # 12~16bit raw image

        org_width = int(dataset_param['width'])
        org_height = int(dataset_param['height'])
    elif camera == 'huateng':
        bitdepth = dataset_param['bitdepth']
        rawfile = np.fromfile(filename, dtype=np.uint16) if bitdepth == 12 else np.fromfile(filename, dtype=np.uint8)
        rawfile = rawfile // 16 if bitdepth == 12 else rawfile #big endian 
        org_width = dataset_param['width']
        org_height = dataset_param['height']
    rawfile = rawfile.reshape(org_height, org_width)
    # crop raw image from original size to Lyncam size
    if crop == True:
        rawfile = crop_to_realBayer(rawfile, org_width, org_height, width, height,crp_x_shft=crp_x_shft, crp_y_shft=crp_y_shft)

    # if output type is 3 channel rgb  (still bayer format)
    if out_type == 'rgb_3ch':
        raw_img_bayer = np.zeros([height, width, 3], dtype=np.uint8)
        for y in range(0, height):
            for x in range(0, width):
                if bayer_type == 0:
                    if(x % 2) == 0 and (y % 2) == 0:
                        raw_img_bayer[y, x, 0] = rawfile[y, x] #red
                    elif (x % 2) == 1 and (y % 2) == 1:
                        raw_img_bayer[y, x, 2] = rawfile[y, x] #blue
                    else:
                        raw_img_bayer[y, x, 1] = rawfile[y, x] #gb, gr
                elif bayer_type == 1:
                    if(x % 2) == 1 and (y % 2) == 0:
                        raw_img_bayer[y, x, 0] = rawfile[y, x] #red
                    elif (x % 2) == 0 and (y % 2) == 1:
                        raw_img_bayer[y, x, 2] = rawfile[y, x] #blue
                    else:
                        raw_img_bayer[y, x, 1] = rawfile[y, x] #gb, gr
        return raw_img_bayer
    #if output is bayer(gray, 1channel)
    elif out_type == 'bayer':
        return rawfile

# Obsoleted, save raw to np array
def save_raw2np(dataset_dir, dataset, camera='mini_ux' ):
    dir = os.path.join(dataset_dir, dataset)
    param_file = os.path.join(dir,dataset) + '.cih'
    param = get_camera_param(param_file)
    width = int(param['width'])
    height = int(param['height'])
    filelist = []
    for i in os.listdir(dir):
        path = os.path.join(dir, i)
        if os.path.isfile(path):
            if(os.path.splitext(path)[1] == '.raw'):
                filelist.append(i)
    for i in tqdm(filelist):
        img_load = os.path.join(dir, i)
        img_save = os.path.join(dir+'_npy', i)
        if os.path.exists(dir+'_npy') == False:
            os.mkdir(dir+'_npy')
        raw_img = load_raw_img(filename = img_load, dataset_param = param, width = width, height = height)

        np.save(img_save, raw_img)
    return


# convert raw bayer to RGB, and save to np array or bmp
def save_raw2selfDemosaic(dataset_dir, dataset, outfiletype = 'np', sensor_width=640, sensor_height=320, crp_x_shft=0, crp_y_shft=0, start=0, length=None, data_save_dir=None, skip=True):
    data_dir = os.path.join(dataset_dir, dataset)
    param_file = os.path.join(data_dir,dataset) + '.cih'
    param = get_camera_param(param_file)
    #width = int(param['width'])
    #height = int(param['height'])
    #bitdepth = int(param['bitdepth'])
    fps = int(param['fps'])
    filelist = []
    for i in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, i)
        if os.path.isfile(path):
            if os.path.splitext(path)[1] == '.raw':
                filelist.append(i)
    
    filelist = filelist[start:]
    if length is not None and length < len(filelist):
        filelist = filelist[:length]
    if fps == 3200 and skip == True:
        filelist = filelist[start::3]
    elif fps == 2000 and skip == True:
        filelist = filelist[start::2]
    cnt = 1
    for i in tqdm(filelist):
        img_load = os.path.join(data_dir, i)
        raw_img = load_raw_img(img_load, param, width=sensor_width, height=sensor_height, bayer_type=0, crp_x_shft=crp_x_shft, crp_y_shft=crp_y_shft)
        #RAW_AWB_torch(bayer=bayer, bayer_pattern='rggb', fixed_gain=raw_awb_gain, bitdepth=raw_bit_depth)
        raw_bitdepth = 12 if raw_img.dtype == 'uint16' else 8
        raw_img[0::2, 0::2] = raw_img[0::2, 0::2] * 0.9
        raw_img[0::2, 1::2] = raw_img[0::2, 1::2] * 0.8
        raw_img[1::2, 0::2] = raw_img[1::2, 0::2] * 0.8
        raw_img[1::2, 1::2] = raw_img[1::2, 1::2] * 1.2
        raw_img = raw_img.clip(min=0, max=(2 ** raw_bitdepth) - 1)
        #if raw_img.dtype == 'uint16':
         #   raw_img = raw_img / 16
         #   raw_img = raw_img.astype(np.uint8)
        ''''''
        #raw_img
        rgb = basic_isp.demosaicing_npy(raw_img, level=1, bitdepth=raw_bitdepth)
        rgb = rgb / 16 if raw_img.dtype == 'uint16' else rgb
        rgb = rgb.astype(np.uint8)
        dir_save = data_dir+'_selfDemosaic' if data_save_dir is None else data_save_dir+'_selfDemosaic'
        dir_save += '_np' if outfiletype == 'np' else '_bmp'
        #print("saving demosaic img to "+dir_save)
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)
        #idx = i.replace(dataset, '')
        strlen=len(str(cnt))
        save_i = '0' * (6-strlen) + str(cnt)
        cnt += 1
        #save_i = str(cnt)
        img_save = os.path.join(dir_save, save_i)
        if outfiletype == 'np':
            img_save = img_save.replace(".raw", "")
            np.save(img_save, rgb)
        else:
            img_save = img_save.replace(".raw", ".bmp")
            image = Image.fromarray(rgb).convert("RGB")
            image.save(img_save)
    return

def save_raw2avi(dataset_dir, dataset, video_write_path, vid_width=640, vid_height=320, crop=False, start=0, length=None, bayer_type='GrR'):
    data_dir = os.path.join(dataset_dir, dataset)
    param_file = os.path.join(data_dir,dataset) + '.cih'
    param = get_camera_param(param_file)
    width = int(param['width'])
    height = int(param['height'])
    save_width = width - 1 if (crop == False) and (bayer_type == 'GrR') else vid_width
    save_height = height if crop is False else vid_height
    save_size = (save_width, save_height)
    out = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc(*'DIVX'), 24, save_size)
    filelist = sorted(os.listdir(data_dir))
    raw_flist = []
    for i in filelist:
        path = os.path.join(data_dir, i)
        if os.path.isfile(path):
            if os.path.splitext(path)[1] == '.raw':
                raw_flist.append(i)
    raw_flist = raw_flist[start:]
    if length is not None and length < len(raw_flist):
        raw_flist = raw_flist[:length]
    for file in tqdm(raw_flist):
        bayer = load_raw_img(os.path.join(data_dir, file), param,
                                 width=save_width, height=save_height, bayer_type=0)

        rgb = basic_isp.demosaicing(bayer, level=1)
        save_img = rgb[..., ::-1] #save as bgr
        #
        out.write(save_img)
    out.release()

    return


def save_bayer(bayer, bit_shft, bayer_pattern):
    bayer_save = bayer / (2 ** bit_shft)
    np.clip(bayer_save, 0, 255)
    bayer_3ch = np.zeros([bayer.shape[0], bayer.shape[1], 3], dtype=np.uint8)
    if bayer_pattern == "rggb":
        bayer_3ch[0::2, 0::2, 0] = bayer_save[0::2, 0::2]
        bayer_3ch[0::2, 1::2, 1] = bayer_save[0::2, 1::2] 
        bayer_3ch[1::2, 0::2, 1] = bayer_save[1::2, 0::2]
        bayer_3ch[1::2, 1::2, 2] = bayer_save[1::2, 1::2]
    elif bayer_pattern == "grbg":
        bayer_3ch[0::2, 1::2, 0] = bayer_save[0::2, 1::2]
        bayer_3ch[0::2, 0::2, 1] = bayer_save[0::2, 0::2]
        bayer_3ch[1::2, 1::2, 1] = bayer_save[1::2, 1::2]
        bayer_3ch[1::2, 0::2, 2] = bayer_save[1::2, 0::2]
    #bayer_3ch = bayer_3ch.clamp(0, 255)
    
    image = Image.fromarray(bayer_3ch).convert("RGB")
    return image

def save_rgb_torch(rgb, name):
    # GPU back to CPU. torch back to numpy
    rgb_out = rgb.cpu()
    rgb_out = rgb_out.numpy()
    rgb_out = rgb_out.astype(np.uint8)
    image = Image.fromarray(rgb_out).convert("RGB")
    image.save(name)
    
def get_camera_param(filename, camera = 'mini_ux'):
    if camera == 'mini_ux':
        data = pd.read_csv(filename, sep = ' : ', dtype=str, engine='python')
        param = {}
        param['fps']  = data.loc['Record Rate(fps)'][0]
        param['shutter'] = data.loc['Shutter Speed(s)'][0]
        param['color_temp'] = data.loc['Color Temperature'][0]
        param['width'] = data.loc['Image Width'][0]
        param['height'] = data.loc['Image Height'][0]
        param['bitdepth'] = data.loc['Color Bit'][0]
    elif camera == 'huateng':
        color_config_org = pd.read_csv(os.path.join(filename, 'color.Config'), sep = ' = ', dtype=str, engine='python')
        mono_config_org = pd.read_csv(os.path.join(filename, 'mono.Config'), sep = ' = ', dtype=str, engine='python')
        param = {}
        param['color'] = {}
        param['color']['fps'] =  1000000 / int(color_config_org.iloc[156, 1].strip(';'))
        param['color']['width'] = int(color_config_org.iloc[15, 1].strip(';'))
        param['color']['height'] = int(color_config_org.iloc[16, 1].strip(';'))
        param['color']['analog_gain'] = float(color_config_org.iloc[56, 1].strip(';')) / 8
        param['color']['exposure_time'] = float(color_config_org.iloc[57, 1].strip(';'))
        param['color']['bitdepth'] = 12 if color_config_org.iloc[73, 1].strip(';') == '1' else 8 if color_config_org.iloc[73, 1].strip(';') == '0' else None
        param['mono'] = {}
        param['mono']['fps'] = 1000000 / int(mono_config_org.iloc[156, 1].strip(';'))
        param['mono']['width'] = int(mono_config_org.iloc[15, 1].strip(';'))
        param['mono']['height'] = int(mono_config_org.iloc[16, 1].strip(';'))
        param['mono']['analog_gain'] = float(mono_config_org.iloc[56, 1].strip(';')) / 8
        param['mono']['exposure_time'] = float(mono_config_org.iloc[57, 1].strip(';'))
        param['mono']['bitdepth'] = 12 if mono_config_org.iloc[73, 1].strip(';') == '1' else 8 if mono_config_org.iloc[73, 1].strip(';') == '0' else None
    return param

#Crop original RAW Bayer image to Lyncam resolution
def crop_to_realBayer(rawfile, org_width = 768, org_height = 512, width=640, height=320, crp_x_shft=0, crp_y_shft=0):

    assert org_height*org_width == rawfile.size
    assert crp_x_shft % 2 == 0 and crp_y_shft % 2 == 0
    if(org_width == width) and (org_height == height):
        return rawfile
    else:
        if org_width == width:
            x_start = 0
        else:
            x_start = int((org_width - width) / 2) + 1 + crp_x_shft
        y_start = int((org_height - height) / 2) + crp_y_shft
        #cropped = np.zeros([height, width], dtype=rawfile.dtype)
        cropped = rawfile[y_start:y_start+height, x_start:x_start+width]
        return cropped


#Crop original RGB image to Lyncam resolution
def crop_RGB(rawfile, org_width = 768, org_height = 512, width=640, height=320, crp_x_shft=0, crp_y_shft=0):
    assert org_height*org_width*3 == rawfile.size
    if(org_width == width) and (org_height == height):
        return rawfile
    else:
        x_start = int((org_width - width) / 2) + 1 + crp_x_shft
        y_start = int((org_height - height) / 2) + crp_y_shft
        #cropped = np.zeros([height, width], dtype=rawfile.dtype)
        cropped = rawfile[y_start:y_start+height, x_start:x_start+width, :]
        return cropped


class Timer:
    def __init__(self, track_num=10, name="default_timer", track_name=None):
        self.record = [[] for i in range(track_num)]
        self.time = None
        self.track = None
        self.name = name
        self.track_name = track_name if track_name is not None else [str(i) for i in range(track_num)]

    def switch(self, track):
        if self.time is None:
            self.time = time()
            self.track = track
            return

        ctime = time()
        dt = ctime - self.time
        self.record[self.track].append(dt)
        self.time = ctime
        self.track = track

    def pause(self):
        dt = time() - self.time
        self.record[self.track].append(dt)
        self.time = None
        self.track = None

    def print(self):
        msg = "Timer: {}\n".format(self.name)
        for i, r in enumerate(self.record):
            if len(r) == 0:
                continue
            msg += "track {}: reps {}, time {:.5f}s\n".format(self.track_name[i], len(r), sum(r))
        print(msg)

    def stop(self):
        self.pause()
        self.print()
