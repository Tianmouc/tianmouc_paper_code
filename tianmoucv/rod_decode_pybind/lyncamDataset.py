import sys
sys.path.append("../lyncamsim_v1.0")
import torch
import torch.utils.data as data
from torch.nn import functional as F
from PIL import Image
import os
import os.path
import random
import time
import numpy as np
from sensorsim_RAW import Lyncam_RAW
from ispsim import LyncamISP


class Tianmouc(data.Dataset):

    def __init__(self, root, original_root, txt_list=['./tianmouc_train.txt','./tianmouc_val.txt'], 
                 transform=None, dim=(640, 320), randomCropSize=(256, 256), train=True, 
                 torch_dev='cuda',nbit_range=[2,3,4,5,6,7,8,9,10,11,12,13,14]):
        self.torch_dev = torch_dev
        
        if self.torch_dev == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.DEFALT_TENSOR = torch.cuda.FloatTensor
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
            self.DEFALT_TENSOR = torch.FloatTensor
        
        Inter_rate  = 30
        self.sensor = Lyncam_RAW(height=dim[1], width=dim[0],
                                 lp_bit_width=8, temp_diff_th=10, spac_diff_th=1, lp_temp_out_coeff=1,
                                    lp_spac_out_coeff=0.8, use_torch=True, torch_dev=self.torch_dev)
        self.rgb_balance = [1,1,1]
        self.train = train
        self.data_list = []
       
        #读取数据描述标签
        if train:
            dataset_info_txt = txt_list[0]
        else:
            dataset_info_txt = txt_list[1]
        frame_sum = 0
        print(dataset_info_txt)
        dataset_info_list = []
        f = open(dataset_info_txt,"r")   #设置文件对象
        line = f.readline()
        line = line[:-1]
        while line:              #直到读取完文件
            line = f.readline()  #读取一行文件，包括换行符
            line = line[:-1]     #去掉换行s符，也可以不去
            data_info = line.split()
            if len(data_info):
                dataset_info_list.append(data_info)
                print(data_info)
        f.close() #关闭文件
        
        #读取数据列表，按视频片段循环
        for label_info in dataset_info_list:
            #对各精度数据做循环
            for nbit in nbit_range:
                dflag  = label_info[0] + str(nbit) + 'bit' #适配多bit混合训练情况
                original  = label_info[1]
                raw_start_frm  = int(label_info[2])
                raw_end_frm  = int(label_info[3])
                flag = True
                if train:
                    convert_root = root  + 'train/' + dflag + '/step_'+str(Inter_rate)+'/'
                else:
                    convert_root = root  + 'val/' + dflag + '/step_'+str(Inter_rate)+'/'
                original_sampleset = original_root  + '/' + original+'/'

                all_files = os.listdir(convert_root + 'rgb')
                file_number = 0
                for each_file in all_files:
                    if not os.path.isdir(each_file):
                        ext = os.path.splitext(each_file)[1] #获取到文件的后缀
                        if ext =='.npy':
                            file_number += 1
                print(dflag ,' has ',file_number,' npy files')
                file_number -= 1 #最后一帧没有后继

                #把同一个插帧目标的训练数据放到同一个dict中去
                for file_id in range(file_number):
                    start_frame = file_id * 30
                    end_frame = (file_id + 1) * 30
                    file_dict = {}

                    #读bayer数据
                    file_dict['start_frame'] = convert_root + 'rgb/' + str(start_frame) + '.npy'
                    file_dict['end_frame'] = convert_root + 'rgb/'   + str(end_frame) + '.npy' 
                    td_frames = []
                    sd_frames = []
                    origin_frames = []

                    #读差分数据
                    #31帧差分，29帧是中间的
                    for inter_id in range(start_frame,end_frame + 1):
                        td_frames.append(convert_root + 'tdiff/' + str(inter_id) + '.npy')
                        sd_frames.append(convert_root + 'sdiff/' + str(inter_id) + '.npy')
                        original_id = inter_id + raw_start_frm + 1#有个id偏置
                        if original_id > raw_end_frm:
                            flag = False
                        strlen=len(str(original_id))
                        filename_format = '0' * (6-strlen) + str(original_id)
                        origin_frames.append(original_sampleset  + filename_format + '.npy')
                    file_dict['td_frames'] = td_frames
                    file_dict['sd_frames'] = sd_frames

                    #存原始reference
                    file_dict['rgb_frames'] = origin_frames
                    file_dict['info'] = [dflag,file_id,nbit]

                    if flag:
                        self.data_list.append(file_dict)

        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root          = root
        self.transform      = transform
        self.train         = train
        self.MAX_FRAME      = 30
        self.Size          = randomCropSize
        self.dim          = dim


    def __getitem__(self, index):

        sample = {}

        if (self.train):
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            IFrameIndex = index % (self.MAX_FRAME-1) + 1
            returnIndex = IFrameIndex
            randomFrameFlip = random.randint(0, 1)
        else:
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            IFrameIndex = index % (self.MAX_FRAME-1) + 1
            returnIndex = IFrameIndex#changed
            randomFrameFlip = 0
        
        # Loop over for all frames corresponding to the `index`.
        file_dict = self.data_list[(index) // (self.MAX_FRAME-1)]#我曹
        nbit = file_dict['info'][2]
        self.isp = LyncamISP(lp_temp_diff_thresh=0, 
                    lp_spac_diff_thresh=0, 
                    lp_temp_out_coeff=1, 
                    lp_spac_out_coeff=1,  #不管原始放缩倍率，统一调为1
                    lp_bit_width=nbit, 
                    torch_dev=self.torch_dev)
      
        # 仿真的rgb图像
        start_frame = self.DEFALT_TENSOR(np.load(file_dict['start_frame']))
        end_frame = self.DEFALT_TENSOR(np.load(file_dict['end_frame']))
        sd0 = self.DEFALT_TENSOR(np.load(file_dict['sd_frames'][0]))
        td0 = self.DEFALT_TENSOR(np.load(file_dict['td_frames'][0]))
        sd1 = self.DEFALT_TENSOR(np.load(file_dict['sd_frames'][self.MAX_FRAME]))
        td1 = self.DEFALT_TENSOR(np.load(file_dict['td_frames'][self.MAX_FRAME]))

        
        # 调用ISP，利用差分信息，对rgb的bayer做填充，用了双线性梯度去马赛克的方法
        self.isp.process(start_frame, td0, sd0, recon_on=False, stat_on=False, flow_on=False)
        start_frame = self.isp.output_buffer['rgb']
        self.isp.process(end_frame, td1, sd1, recon_on=False, stat_on=False, flow_on=False)
        end_frame = self.isp.output_buffer['rgb']

        # 原始的rgb图像作为标签
        inter_frame = self.DEFALT_TENSOR(np.load(file_dict['rgb_frames'][IFrameIndex]))
        start_frame_raw = self.DEFALT_TENSOR(np.load(file_dict['rgb_frames'][0]))
        end_frame_raw = self.DEFALT_TENSOR(np.load(file_dict['rgb_frames'][self.MAX_FRAME])  )  
        
        # white balance
        #inter_frame = self._wight_balance(inter_frame)
        #start_frame_raw = self._wight_balance(start_frame_raw)
        #end_frame_raw = self._wight_balance(end_frame_raw)
        
        # 时间差分和空间差分数据
        tdiff_all = torch.zeros([self.MAX_FRAME+1, self.dim[1]//2,self.dim[0]//2])
        sdiff_all = torch.zeros([self.MAX_FRAME+1, self.dim[1]//2,self.dim[0]//2, 4])
        stdiff_frames = torch.zeros([self.MAX_FRAME+1, 3, self.Size[1],self.Size[0]])
        
        #读入数据：diff值
        for i in range(self.MAX_FRAME+1):
            # 原始数据，原始大小
            # s: 只有奇数行奇数列大像素位置有四个差值，其他大像素、小像素位置都没有值；t: 奇数行奇数列和偶数行偶数列的大像素有值
            td_frame = np.load(file_dict['td_frames'][i])
            sd_frame = np.load(file_dict['sd_frames'][i])
            # ISP处理,包含了阈值恢复和系数缩放
            # s: 奇数行奇数列有值, 160*320； t: 所有数值都填充完毕, 160*320
            tdiff_all[i,...] = self.isp.process_tdiff(self.DEFALT_TENSOR(td_frame))
            sdiff_all[i,...] = self.isp.process_sdiff(self.DEFALT_TENSOR(sd_frame))
        diff_flow = torch.zeros([self.MAX_FRAME+1,3, tdiff_all.size(1)* 2, tdiff_all.size(2) * 2])
        down_flow, right_flow = self.isp.get_edge_flow(sdiff_all)
        
        
        #数值范围是-255~255吗？
        #different bit quantization noramlization
        diff_flow[:, 0, 0::2, 0::2] = diff_flow[:, 0,0::2, 1::2] = \
                diff_flow[:, 0,1::2, 0::2] = diff_flow[:, 0,1::2, 1::2] = tdiff_all
        diff_flow[:, 1, 0::2, 0::2] = diff_flow[:, 1,0::2, 1::2] = \
                diff_flow[:, 1,1::2, 0::2] = diff_flow[:, 1,1::2, 1::2] = down_flow
        diff_flow[:, 2, 0::2, 0::2] = diff_flow[:, 2,0::2, 1::2] = \
                diff_flow[:, 2,1::2, 0::2] = diff_flow[:, 2,1::2, 1::2] = right_flow
        
        for i in range(self.MAX_FRAME):
            diff_flow_tf = self._transformer(diff_flow[i,...].permute(1,2,0), cropArea=cropArea, frameFlip=randomFrameFlip)
            stdiff_frames[i,...] = diff_flow_tf.permute(2,0,1)
        #随机裁剪
        start_frame = self._transformer(start_frame, cropArea=cropArea, frameFlip=randomFrameFlip)
        end_frame = self._transformer(end_frame, cropArea=cropArea, frameFlip=randomFrameFlip)
        start_frame_raw = self._transformer(start_frame_raw, cropArea=cropArea, frameFlip=randomFrameFlip)
        end_frame_raw = self._transformer(end_frame_raw, cropArea=cropArea, frameFlip=randomFrameFlip)
        inter_frame = self._transformer(inter_frame, cropArea=cropArea, frameFlip=randomFrameFlip)

        #存进单个插帧训练字典中,全部归一化后备用
        # 归一化至0~1
        sample['F0'] = start_frame/255.0
        sample['F1'] = end_frame/255.0
        sample['F0_raw'] = start_frame_raw/255.0
        sample['F1_raw'] = end_frame_raw/255.0
        sample['Ft'] = inter_frame/255.0
        sample['tsdiff'] = stdiff_frames/255.0
        sample['dflag'] = file_dict['info'][0]
        return sample, returnIndex

    def __len__(self):
        return len(self.data_list*(self.MAX_FRAME-1))

    def _transformer(self, img, cropArea=None, frameFlip=0):
        if (cropArea != None):
            x1,y1,x2,y2 = cropArea
            img = img[y1:y2,x1:x2,:]
        if frameFlip:
            inverse_img = img.cpu().numpy()[::-1,...]
            img = self.DEFALT_TENSOR(inverse_img.copy())
        return img

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    