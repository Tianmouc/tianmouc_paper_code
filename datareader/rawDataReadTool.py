import numpy as np
import os
import struct
import cv2,sys
import matplotlib.pyplot as plt
import torch
from lib.utils import lyncam_raw_comp
from lib.basic_isp import demosaicing_npy

import scripts.rod_decoder_py as rdc
#from lib.utils import lyncam_raw_comp
#from lib.basic_isp import demosaicing_npy

adc_bit_prec = 8
width = 160
height = 160
#interface =
ROD_8B_ONE_FRM = 0x9e00       #158KB * 1024 / 4;//0x9e00
ROD_4B_ONE_FRM = 0x4D00 
ROD_2B_ONE_FRM = 0x1C40
CONE_TYPE = 0
ROD_TYPE  = 1

def findDataNum(filename):
    data_info = filename.split('/')[-1]
    data_info = data_info.split('.')[0]
    num_stamp = data_info.split('_')
    return int(num_stamp[0])

def findDataTimestamp(filename):
    data_info = filename.split('/')[-1]
    data_info = data_info.split('.')[0]
    num_stamp = data_info.split('_')
    return int(num_stamp[1])
    
    

    
class TianmoucDataRead():
    
    def __init__(self,rod_adc_bit, dataset_top = "/data/lyncam_data/Lyncam",ext=".bin",rodfilepersample = 2,showList=True):
    
        ext=".bin"
        timeThresh = 1000#10ms?
        fileDict = dict([])
        signals= ['rod','cone']
        self.rod_height = 160
        self.rod_width = 160
        self.cone_height = 320
        self.cone_width = 320
        self.rod_adc_bit = rod_adc_bit
        self.rod_img_per_file = rodfilepersample
        self.rodfilepersample = rodfilepersample

        for folderlist in os.listdir(dataset_top):
            firstSave = True
            folder_root = os.path.join(dataset_top, folderlist)
            for sg in signals:
                try:
                    fileList = []
                    dataListRoot = os.path.join(folder_root, sg)
                    for fl in os.listdir(dataListRoot):
                        flpath = os.path.join(dataListRoot, fl)
                        if (os.path.isfile(flpath) and flpath[-4:]==ext):
                            fileList.append(flpath)
                    if len(fileList)>0:
                        if firstSave == True:
                            fileDict[folderlist] = {}
                            fileDict[folderlist][sg] = fileList
                            firstSave = False
                        else:
                            fileDict[folderlist][sg] = fileList
                    elif folderlist in fileDict:
                        fileDict.pop(folderlist)
                        break
                except:
                    print('bad entry for ',sg,' in ',folder_root)

        keylist = [k for k in fileDict]
        for key in keylist:
            for k in fileDict[key]:
                if showList:
                    print(key,k,len(fileDict[key][k]))
                list2 = sorted(fileDict[key][k],key=findDataNum)
                fileDict[key][k] = list2


        for key in keylist:
            try:
                rodListSorted = fileDict[key]['rod']
                coneListSorted = fileDict[key]['cone']
            except:
                fileDict.pop(key)
                continue

            minGap = 1e20
            coneStartID = -1
            rodStartID = -1
            for coneID in range(len(coneListSorted)):
                if minGap < 200:
                    break
                conetimestamp = findDataTimestamp(coneListSorted[coneID])
                for rodID in range(len(rodListSorted)):
                    rodtimestamp = findDataTimestamp(rodListSorted[rodID])
                    gap = abs(conetimestamp-rodtimestamp)
                    if gap < minGap:
                        minGap = gap
                        coneStartID = coneID
                        rodStartID = rodID
                        if minGap < timeThresh:
                            break
            coneEndID = -1
            rodEndID = -1    
            minGap = 1e20
            for coneID in range(len(coneListSorted)-1,-1,-1):
                if minGap < 200:
                    break
                conetimestamp = findDataTimestamp(coneListSorted[coneID])
                for rodID in range(len(rodListSorted)-1,-1,-1):
                    rodtimestamp = findDataTimestamp(rodListSorted[rodID])
                    gap = abs(conetimestamp-rodtimestamp)
                    if gap < minGap:
                        minGap = gap
                        coneEndID = coneID
                        rodEndID = rodID
                        if minGap < timeThresh:
                            break
            fileDict[key]['pair'] = [(coneStartID,rodStartID),(coneEndID,rodEndID)]
        self.fileDict = fileDict
        self.rodfilepersample = rodfilepersample
        
    def dataNum(self,key):
        return self.fileDict[key]['pair'][1][0] - self.fileDict[key]['pair'][0][0] 
        

    def readFile(self,key,cone_id,rod_id,viz=True,useISP=False,ifSync =True):
        
        coneIdStart = 0
        rodIdStart = 0
        if ifSync:
            
            coneIdStart = self.fileDict[key]['pair'][0][0]
            rodIdStart = self.fileDict[key]['pair'][0][1]
            
        rodFileID  = rodIdStart + (rod_id)//2
        filename = self.fileDict[key]['rod'][rodFileID]
        size = os.path.getsize(filename)
        rodtimeStamp = findDataTimestamp(filename)
            
        pvalue_np           = np.zeros(size // 4 * self.rodfilepersample, dtype=np.int32)
        temp_diff_np        = np.zeros(width * height, dtype=np.int8)
        spat_diff_left_np   = np.zeros(width * height, dtype=np.int8)
        spat_diff_right_np  = np.zeros(width * height, dtype=np.int8)

        idbias = ((rodIdStart*2 + rod_id) % self.rodfilepersample) * 40448
        
        rodtimeStamp += ((rodIdStart*2 + rod_id) % self.rodfilepersample) * 130
        with open(filename, 'rb') as f:
            for i in range(40448 * 2):
                data = f.read(4)
                realdata = struct.unpack('i',data)
                data_int = realdata[0]
                if idbias <= i and i < idbias+40448:
                    pvalue_np[i-idbias] = data_int
                
        ret_code = rdc.rod_decoder_py(pvalue_np, temp_diff_np, spat_diff_left_np, spat_diff_right_np, width, height)
        temp_diff_np = np.reshape(temp_diff_np, (width, height))
        spat_diff_left_np = np.reshape(spat_diff_left_np, (width, height))
        spat_diff_right_np = np.reshape(spat_diff_right_np, (width, height))


        conefilename = self.fileDict[key]['cone'][coneIdStart + cone_id]
        size = os.path.getsize(conefilename)
        pvalue_np = np.zeros(size // 4, dtype=np.int32)
        wihei   = int(np.sqrt((size-64)//4))
        data_np = np.zeros((size-64)//4 ,dtype=np.int32)

        conetimeStamp = findDataTimestamp(conefilename)
        with open(conefilename, 'rb') as f:
            for i in range(size//4):
                data = f.read(4)
                realdata = struct.unpack('i',data)
                data_int = realdata[0]
                pvalue_np[i] = data_int

        data_np = pvalue_np[16:]
        cone_raw = np.reshape(data_np, (wihei, wihei))
        torch_dev='cpu'

        if torch_dev == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            DEFALT_TENSOR = torch.cuda.FloatTensor
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
            DEFALT_TENSOR = torch.FloatTensor

        #start_frame = DEFALT_TENSOR(cone_raw)
        sdl = DEFALT_TENSOR(spat_diff_left_np)
        sdr = DEFALT_TENSOR(spat_diff_right_np)
        td = DEFALT_TENSOR(temp_diff_np)
        
        sd = None
        
        #if useISP:
        #    sd = torch.stack([sdl,sdr],dim=-1)
        #else:
        sd = torch.stack([sdl,sdr],dim=-1)
    
        rgb_processed = None

        #if useISP:
        #    rgb_processed = isp.demosaicing_bilinear_gradient_correct(start_frame, sd)
        #else:
        #rgb_processed = 
        rgb_processed = lyncam_raw_comp(cone_raw)
        rgb_processed = demosaicing_npy(rgb_processed, 'bggr', 1, 10)
        
        if viz:
            plt.figure(figsize=(5,5)) 
            plt.subplot(2,2,1)
            plt.imshow(temp_diff_np)
            plt.subplot(2,2,2)
            plt.imshow(spat_diff_left_np)
            plt.subplot(2,2,3)
            plt.imshow(spat_diff_right_np)
            plt.subplot(2,2,4)
            plt.imshow(rgb_processed/1024)
            plt.show()
        
        return rgb_processed,td,sd,rodtimeStamp,conetimeStamp,conefilename
    

    def readFileFast(self,key,cone_id,rod_id,viz=True,useISP=False,ifSync =True):
        coneIdStart = 0
        rodIdStart = 0
        if ifSync:
            coneIdStart = self.fileDict[key]['pair'][0][0]
            rodIdStart = self.fileDict[key]['pair'][0][1]

        rodFileID = rodIdStart + (rod_id) #// self.rod_img_per_file
        filename = self.fileDict[key]['rod'][rodFileID]

        rodtimeStamp = findDataTimestamp(filename)
        #print(rodtimeStamp)

        bytesize = os.path.getsize(filename)
        size = bytesize // 4

        temp_diff_np = np.zeros((self.rod_img_per_file, self.rod_width * self.rod_height), dtype=np.int8)
        spat_diff_left_np = np.zeros((self.rod_img_per_file, self.rod_width * self.rod_height), dtype=np.int8)
        spat_diff_right_np = np.zeros((self.rod_img_per_file, self.rod_width * self.rod_height), dtype=np.int8)
        pkt_size_np = np.zeros([self.rod_img_per_file], dtype=np.int32)
        pkt_size_td = np.zeros([self.rod_img_per_file], dtype=np.int32)
        pkt_size_sd = np.zeros([self.rod_img_per_file], dtype=np.int32)
        one_frm_size = size // self.rod_img_per_file
        ret_code = rdc.rod_decoder_py_byfile_td_sd_bw(filename, self.rod_img_per_file, size, one_frm_size,
                                                temp_diff_np, spat_diff_left_np, spat_diff_right_np,
                                                pkt_size_np,pkt_size_td,pkt_size_sd,
                                                self.rod_height, self.rod_width)
        sd_list = []
        td_list = []
        for b in range(self.rod_img_per_file):
            width = self.rod_width
            height = self.rod_height
            temp_diff_np1 = np.reshape(temp_diff_np[b, ...], (width, height))
            spat_diff_left_np1 = np.reshape(spat_diff_left_np[b, ...], (width, height))
            spat_diff_right_np1 = np.reshape(spat_diff_right_np[b,...], (width, height))

            sdl = spat_diff_left_np1
            sdr = spat_diff_right_np1
            td = temp_diff_np1
            sd_list.append((sdl, sdr))
            td_list.append(td)

        conefilename = self.fileDict[key]['cone'][coneIdStart + cone_id]
        conetimeStamp = findDataTimestamp(conefilename)
        size = os.path.getsize(conefilename)
        #wihei = int(np.sqrt((size - 64) // 4))
        raw_vec = np.zeros(self.cone_height * self.cone_width, dtype=np.int16)
        rdc.cone_reader_py_byfile(conefilename, size // 4, raw_vec, self.cone_height, self.cone_width)
        cone_raw = np.reshape(raw_vec, (self.cone_height, self.cone_width))
        start_frame = cone_raw

        #rgb_processed = start_frame
        start_frame = start_frame.astype(np.float32)
        
        return start_frame, td_list, sd_list, pkt_size_np, rodtimeStamp, conetimeStamp,pkt_size_td,pkt_size_sd
    

    def cone_read_single(self, conefilename):
        conetimeStamp = findDataTimestamp(conefilename)
        size = os.path.getsize(conefilename)
        wihei = int(np.sqrt((size - 64) // 4))
        raw_vec = np.zeros(self.cone_height * self.cone_width, dtype=np.int16)
        cone_img_timestamp = np.zeros([1], dtype=np.uint64)
        cone_img_fcnt = np.zeros([1], dtype=np.int32)
        cone_img_adcprec = np.zeros([1], dtype=np.int32)
        rdc.cone_reader_py_fullInfo(conefilename, size // 4, raw_vec, 
                                    cone_img_timestamp, cone_img_fcnt, cone_img_adcprec,
                                    self.cone_height, self.cone_width)
        cone_raw = np.reshape(raw_vec, (self.cone_height, self.cone_width))
        cone_timestamp_real = cone_img_timestamp[0]
        cone_fcnt_real = cone_img_fcnt[0]
        return cone_fcnt_real, cone_timestamp_real, cone_raw

    def rod_read_single(self, filename, only_info = False, full_info = False,filesize_correct=False, correct_size=237568):
        rodtimeStamp = findDataTimestamp(filename)
        if filesize_correct:
            bytesize = correct_size
        else:
            bytesize = os.path.getsize(filename)
        size = bytesize // 4
        one_frm_size = size // self.rod_img_per_file

        rod_img_timestamp_np = np.zeros([self.rod_img_per_file], dtype=np.uint64)
        rod_fcnt_np = np.zeros([self.rod_img_per_file], dtype=np.int32)
        rod_adcprec_np = np.zeros([self.rod_img_per_file], dtype=np.int32)
        if only_info:
            ret_code = rdc.rod_decoder_py_onlyinfo(filename, self.rod_img_per_file, size, one_frm_size,
                                                rod_img_timestamp_np, rod_fcnt_np, rod_adcprec_np,
                                                self.rod_height, self.rod_width)
            return rod_img_timestamp_np, rod_fcnt_np, rod_adcprec_np
        else:
            temp_diff_np = np.zeros((self.rod_img_per_file, self.rod_width * self.rod_height), dtype=np.int8)
            spat_diff_left_np = np.zeros((self.rod_img_per_file, self.rod_width * self.rod_height), dtype=np.int8)
            spat_diff_right_np = np.zeros((self.rod_img_per_file, self.rod_width * self.rod_height), dtype=np.int8)
            pkt_size_np = np.zeros([self.rod_img_per_file], dtype=np.int32)
            pkt_size_td = np.zeros([self.rod_img_per_file], dtype=np.int32)
            pkt_size_sd = np.zeros([self.rod_img_per_file], dtype=np.int32)            
            # ret_code = rdc.rod_decoder_py(pvalue_np, temp_diff_np, spat_diff_left_np, spat_diff_right_np, width, height)
            ret_code = rdc.rod_decoder_py_fullInfo(filename, self.rod_img_per_file, size, one_frm_size,
                                                    temp_diff_np, spat_diff_left_np, spat_diff_right_np,
                                                    pkt_size_np, pkt_size_td, pkt_size_sd,
                                                    rod_img_timestamp_np, rod_fcnt_np, rod_adcprec_np,
                                                    self.rod_height, self.rod_width)
            if full_info:
                return temp_diff_np, spat_diff_left_np, spat_diff_right_np, pkt_size_np,pkt_size_td,pkt_size_sd, rod_img_timestamp_np, rod_fcnt_np, rod_adcprec_np
            else:
                return temp_diff_np, spat_diff_left_np, spat_diff_right_np, pkt_size_np,pkt_size_td,pkt_size_sd,
    
    def readFileSeq(self, key, cone_start, cone_duration, rod_cone_ratio, ifSync =False, filesize_correct=False, correct_size=237568):
        """Read cone (COP) and rod (AOP) sequentially

        Args:
            key (string): the dataset name
            cone_start (int): the start index of cone
            cone_duration (int): the duration of cone want to read
            rod_cone_ratio (int): rod speed / cone speed
            ifSync (bool, optional): _description_. Defaults to True.

        Yields:
            _type_: generate cone or rod read list
        """        
        coneIdStart_sync = 0
        #rodIdStart_sync = 0
        if ifSync:
            coneIdStart_sync = self.fileDict[key]['pair'][0][0]
            #rodIdStart_sync = self.fileDict[key]['pair'][0][1]
        cone_start_id = coneIdStart_sync + cone_start
        # read one cone
        if cone_duration > len(self.fileDict[key]['cone']) - cone_start_id:
            cone_duration = len(self.fileDict[key]['cone']) - cone_start_id
        rodListSorted = self.fileDict[key]['rod']
        coneListSorted = self.fileDict[key]['cone']
        #Sync the rod and cone for initialization
        for rodID in range(len(rodListSorted)):
            rodtimestamp = findDataTimestamp(rodListSorted[rodID])
            gap = findDataTimestamp(coneListSorted[cone_start_id])-rodtimestamp
            if gap < 0 :
                rodFileID_start = rodID
                break
        rodFileID = rodFileID_start #
        
        for ci in range(cone_start_id, cone_start_id + cone_duration):
            conefilename = self.fileDict[key]['cone'][ci]        
            cone_fcnt_real, cone_timestamp_real, cone_raw = self.cone_read_single(conefilename)
             # yield current Cone frame
            yield [CONE_TYPE, cone_fcnt_real, cone_timestamp_real, cone_raw]
            # find next timestamp
            if ci < cone_start_id + cone_duration - 1:    
                _, cone_timestamp_next, _ = self.cone_read_single(self.fileDict[key]['cone'][ci + 1])
            else:
                cone_timestamp_next = cone_timestamp_real + 3267
            rod_in_cycle = True
            if rodFileID >= len(self.fileDict[key]['rod']):
                break
            # read rod data between current cone and next cone
            while(rod_in_cycle):
                if rodFileID >= len(self.fileDict[key]['rod']):
                    break
                rod_filename = self.fileDict[key]['rod'][rodFileID]
                rod_in_range = []
                temp_diff_np, spat_diff_left_np, spat_diff_right_np, pkt_size_np, \
                pkt_size_td,pkt_size_sd, rod_img_timestamp_np, rod_fcnt_np, rod_adcprec_np \
                    = self.rod_read_single(filename=rod_filename, only_info=False, full_info=True,
                                           filesize_correct=filesize_correct, correct_size=correct_size)
                # if rod timestamp >= current cone and < next cone, read out it
                for rp in range(self.rod_img_per_file):
                    if rod_img_timestamp_np[rp] >= cone_timestamp_real and  rod_img_timestamp_np[rp] < cone_timestamp_next:
                        rod_in_range.append(True)
                    else:
                        rod_in_range.append(False)
                
                if not any(rod_in_range):
                    rod_in_cycle = False
                    rodFileID = rodFileID
                    break
                for b in range(self.rod_img_per_file):
                    if rod_in_range[b]:
                        width = self.rod_width
                        height = self.rod_height
                        temp_diff_np1 = np.reshape(temp_diff_np[b, ...], (width, height))
                        spat_diff_left_np1 = np.reshape(spat_diff_left_np[b, ...], (width, height))
                        spat_diff_right_np1 = np.reshape(spat_diff_right_np[b, ...], (width, height))
                        rod_fcnt_np1 = rod_fcnt_np[b]
                        rod_img_timestamp_np1 = rod_img_timestamp_np[b]
                        rod_adcprec_np1 = rod_adcprec_np[b]
                        pkt_size_np1 = pkt_size_np[b]
                        pkt_size_sd1 = pkt_size_sd[b]
                        pkt_size_td1 = pkt_size_td[b]
                        # yield current rod frames: TD, SDL, SDR
                        yield [ROD_TYPE,rod_fcnt_np1, rod_img_timestamp_np1,rod_adcprec_np1,temp_diff_np1,spat_diff_left_np1,spat_diff_right_np1, pkt_size_np1, pkt_size_sd1, pkt_size_td1]  
                  
                if all(rod_in_range) or rod_in_range[0] is False:
                    rod_in_cycle = True
                    rodFileID += 1
                #idbias = ((rodIdStart * 2 + rod_id) % self.rodfilepersample) * 40448
                #rodtimeStamp += ((rodIdStart * 2 + rod_id) % self.rodfilepersample) * 130
                else:
                    rod_in_cycle = False
                    rodFileID = rodFileID