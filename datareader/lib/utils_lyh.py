import numpy as np
import cv2
import os
from time import time
import imageio
import pandas as pd
from tqdm import tqdm
from scipy.signal import convolve2d
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(__file__))
import lib.basic_isp as basic_isp

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


def load_from_folder(folder_pth, dataset, start=0, length=None, filetype='selfDemoaic_np', sensor_height=320, sensor_width=640):
    '''
    Call this function to load dataset from folder

    :param folder_pth: dataset folder path
    :param dataset: dataset name
    :param start: frane start
    :param length: stop = start + length
    :param filetype: raw--raw bayer from camera, selfDemoaic_np: self demosaicing RGB np array,
    :param sensor_height: sensor height
    :param sensor_width: sensor width
    :return:None
    '''
    file_list = sorted(os.listdir(folder_pth))[start:]
    raw_filelist = []
    if filetype == 'raw':
        for i in file_list:
            path = os.path.join(folder_pth, i)
            if os.path.isfile(path):
                if os.path.splitext(path)[1] == '.raw':
                    raw_filelist.append(i)
    #raw_filelist = raw_filelist[start:]

    if length is not None and length < len(file_list):
        file_list = file_list[:length]
        raw_filelist = raw_filelist[:length]
    # raw image: first convert to RGB
    if filetype == 'raw':
        param_file = os.path.join(folder_pth, dataset) + '.cih'
        dataset_param = get_camera_param(param_file)
        for file in raw_filelist:
            bayer = load_raw_img(os.path.join(folder_pth, file), dataset_param,
                                 width=sensor_width, height=sensor_height, bayer_type=0)
            img = basic_isp.demosaicingf(bayer, level=1)
            yield img
            #return img
    elif filetype == 'selfDemoaic_np':
        for file in file_list:
            img = np.load(os.path.join(folder_pth, file))
            yield img
    else:
        for file in file_list:
            img = cv2.imread(os.path.join(folder_pth, file))[..., ::-1]  # flip from BGR to RGB
            yield img
            #return img


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
def load_raw_img(filename, dataset_param, width=640, height=320, bayer_type=0, out_type = 'bayer', crp_x_shft=0, crp_y_shft=0):
    """
    Call this function to load raw bayer image
    :param filename: raw image to be loaded
    :param dataset_param: dataset parameters
    :param width: Lyncam real width
    :param height: Lyncam real height
    :param bayer_type: 0--RGrRGr...GbBGbB, 1--GrRGrR...BGbBGb...;
    :param out_type: 'bayer', 1channel bayer array(grayscale); 'rgb_3ch', RGB 3 channel Bayer array
    """
    bitdepth = int(dataset_param['bitdepth']) # bit depth of the sensor raw data(8, 16...)
    assert (bitdepth == 8) or (bitdepth == 16)
    if bitdepth == 8:
        rawfile = np.fromfile(filename, dtype=np.uint8)  # 8bit raw image
    elif bitdepth == 16:  # 16bit per pixel, real depth: 12~16bit; if 12bit, 'h0XXX, X is effective pixel value
        rawfile = np.fromfile(filename, dtype=np.int16)  # 12~16bit raw image

    org_width = int(dataset_param['width'])
    org_height = int(dataset_param['height'])
    rawfile = rawfile.reshape(org_height, org_width)
    # crop raw image from original size to Lyncam size
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
def save_raw2selfDemosaic(dataset_dir, dataset, outfiletype = 'bmp', sensor_width=640, sensor_height=320, crp_x_shft=0, crp_y_shft=0, start=0, length=None):
    data_dir = os.path.join(dataset_dir, dataset)
    param_file = os.path.join(data_dir,dataset) + '.cih'
    param = get_camera_param(param_file)
    width = int(param['width'])
    height = int(param['height'])
    depth = int(param['bitdepth'])
    filelist = []
    for i in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, i)
        if os.path.isfile(path):
            if os.path.splitext(path)[1] == '.raw':
                filelist.append(i)
    filelist = filelist[start:]
    if length is not None and length < len(filelist):
        filelist = filelist[:length]
    for i in tqdm(filelist):
        img_load = os.path.join(data_dir, i)
        raw_img = load_raw_img(img_load, param, width=sensor_width, height=sensor_height, bayer_type=0, crp_x_shft=crp_x_shft, crp_y_shft=crp_y_shft)
        rgb = basic_isp.demosaicing(raw_img, level=1)
        dir_save = data_dir+'_selfDemosaic'
        dir_save += '_np' if outfiletype == 'np' else '_bmp'
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)
        save_i = i.replace(dataset, '')
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
    save_width = width - 1 if (crop is False) and (bayer_type is 'GrR') else vid_width
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

def get_camera_param(filename):
    data = pd.read_csv(filename, sep = ' : ', dtype=str)
    param = {}
    param['fps']  = data.loc['Record Rate(fps)'][0]
    param['shutter'] = data.loc['Shutter Speed(s)'][0]
    param['color_temp'] = data.loc['Color Temperature'][0]
    param['width'] = data.loc['Image Width'][0]
    param['height'] = data.loc['Image Height'][0]
    param['bitdepth'] = data.loc['Color Bit'][0]
    return param

#Crop original RAW Bayer image to Lyncam resolution
def crop_to_realBayer(rawfile, org_width = 768, org_height = 512, width=640, height=320, crp_x_shft=0, crp_y_shft=0):

    assert org_height*org_width == rawfile.size
    assert crp_x_shft % 2 == 0 and crp_y_shft % 2 == 0
    if(org_width == width) and (org_height == height):
        return rawfile
    else:
        x_start = int((org_width - width) / 2) + 1 + crp_x_shft
        y_start = int((org_height - height) / 2) + crp_y_shft
        #cropped = np.zeros([height, width], dtype=rawfile.dtype)
        cropped = rawfile[y_start:y_start+height, x_start:x_start+width]
        return cropped

#Crop original RGB image to Lyncam resolution
def crop_RGB(rawfile, org_width = 768, org_height = 512, width=640, height=320):
    assert org_height*org_width*3 == rawfile.size
    if(org_width == width) and (org_height == height):
        return rawfile
    else:
        x_start = int((org_width - width) / 2) + 1
        y_start = int((org_height - height) / 2)
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