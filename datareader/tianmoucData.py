import numpy as np
import os
import struct
import cv2,sys
import matplotlib.pyplot as plt
import torch
import scripts.rod_decoder_py as rdc
import torch.nn.functional as F
from lib.utils import lyncam_raw_comp
from lib.basic_isp import demosaicing_npy

sys.path.append("../")
flag = True
try:
    from tianmoucv.basic import fourdirection2xy,poisson_blend
except:
    flag = False

adc_bit_prec = 8
width = 160
height = 160
#interface =
ROD_8B_ONE_FRM = 0x9e00       #158KB * 1024 / 4;//0x9e00
ROD_4B_ONE_FRM = 0x4D00 
ROD_2B_ONE_FRM = 0x1C40

_DEBUG = True

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
    
class TianmoucDataReader():
    def addMoreSample(self,fileDict,dataset_top,MAXLEN,matchkey):
        signals= ['rod','cone']
        ext=".bin"
        timeThresh = 1000#10ms?
        for folderlist in os.listdir(dataset_top):
            if matchkey is not None and folderlist!=matchkey:
                continue
            
            #print('add:',dataset_top,'s keys:',folderlist)
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
                            fileDict[folderlist+'@'+dataset_top] = {}
                            fileDict[folderlist+'@'+dataset_top][sg] = fileList
                            firstSave = False
                        else:
                            fileDict[folderlist+'@'+dataset_top][sg] = fileList
                    elif folderlist in fileDict:
                        fileDict.pop(folderlist)
                        break
                except:
                    continue
                    print('bad entry for ',sg,' in ',folder_root)
        
        keylist = [key for key in fileDict]
                
        for key in keylist:
            for sg in signals:
                list2 = sorted(fileDict[key][sg],key=findDataNum)
                fileDict[key][sg] = list2
                

        MINTIMEINTERVAL = 131
        for key in keylist:
            legalFileList = []
            try:
                rodListSorted = fileDict[key]['rod']
                coneListSorted = fileDict[key]['cone']
            except:
                fileDict.pop(key)
                print('key:',key,' has some problems, popped')
                continue
                
            #search for the rodID who meet the requirements
            if len(rodListSorted) > 100:
                MINTIMEINTERVAL = (findDataTimestamp(rodListSorted[100]) - 
                               findDataTimestamp(rodListSorted[0]))//(100 * 2) #下界
            if _DEBUG:
                print('MINTIMEINTERVAL:',MINTIMEINTERVAL,' len:',len(rodListSorted))
                
            rodStart = 0
            for coneID in range(len(coneListSorted)-1):
                legalFileDict = dict([])
                conetimestamp1 = findDataTimestamp(coneListSorted[coneID])
                conetimestamp2 = findDataTimestamp(coneListSorted[coneID+1])
                rodtimestamp1 = -500
                rodtimestamp2 = -500
                rodRange = [-1,-1]
                coneInterval = conetimestamp2-conetimestamp1
                
                if _DEBUG:
                    if abs(coneInterval)> 3300:
                        print(key)
                        print('CONE INTERVL:',conetimestamp2-conetimestamp1)
                        print(coneListSorted[coneID-2:coneID+2])
                        
                # gurantee
                for rodID in range(rodStart,len(rodListSorted)):
                    rodtimestamp1 = findDataTimestamp(rodListSorted[rodID])
                    deltaT = rodtimestamp1-conetimestamp1
                    if deltaT > 0:
                        if deltaT > MINTIMEINTERVAL*2+2:
                            rodRange[0] = -1
                            if _DEBUG and rodID>0:
                                print('=========first======')
                                print('rodID:',rodID)
                                print('rodID:',key)
                        elif deltaT>= MINTIMEINTERVAL*2-2 and deltaT <= MINTIMEINTERVAL*2+2:
                            rodRange[0] = rodID*self.rodfilepersample - 2
                        elif deltaT>= MINTIMEINTERVAL*1-2 and deltaT <= MINTIMEINTERVAL*1+2:
                            rodRange[0] = rodID*self.rodfilepersample - 1
                        else:
                            rodRange[0] = rodID*self.rodfilepersample 
                        break 
                
                #next scan
                rodStart = max(rodRange[0]//self.rodfilepersample-1,0)
                if rodRange[0] < 0 or coneInterval>3300:
                    continue
                
                    
                for rodID in range(rodStart,len(rodListSorted)):
                    rodtimestamp2 = findDataTimestamp(rodListSorted[rodID])
                    deltaT = rodtimestamp2-conetimestamp2
                    if deltaT > 0:
                        if deltaT > MINTIMEINTERVAL*2+2:
                            rodRange[0] = -1
                            if _DEBUG and rodID>0:
                                print('=========second======')
                                print('rodID:',rodID)
                                print('rt-5~rt-5',rodListSorted[max(0,rodID-5):min(len(rodListSorted),rodID+5)],
                                          ' ct:',conetimestamp2)
                                print('deltaT',deltaT)
                                print('last DeltaT:',findDataTimestamp(rodListSorted[rodID-1])-conetimestamp2)
                        elif deltaT>= MINTIMEINTERVAL*2-2 and deltaT <= MINTIMEINTERVAL*2+2:
                            rodRange[1] = rodID*self.rodfilepersample - 2
                        elif deltaT>= MINTIMEINTERVAL*1-2 and deltaT <= MINTIMEINTERVAL*1+2:
                            rodRange[1] = rodID*self.rodfilepersample - 1
                        else:
                            rodRange[1] = rodID*self.rodfilepersample 
                        break 
                        
                rodRange[1] += 1
                rodStart = max(rodRange[1]//self.rodfilepersample-1,0)
                #add training data
                
                if rodRange[1] < rodRange[0]+25:
                    continue
                legalFileDict['cone'] = [coneID,coneID+1]
                legalFileDict['rod'] = rodRange
                legalFileList.append(legalFileDict)
                if MAXLEN>0 and len(legalFileList)>MAXLEN:
                    break

            fileDict[key]['rodfilepersample'] = 2 #TODO
            fileDict[key]['trainData'] = legalFileList
        
        print(keylist,[key for key in fileDict])
                
        return fileDict  
              

    def __init__(self,pathList,showList=True,rod_adc_bit=8,MAXLEN=-1,matchkey=None):
        self.rod_height = 160
        self.rod_width = 160
        self.showList = showList
        self.cone_height = 320
        self.cone_width = 320
        self.rod_adc_bit = rod_adc_bit
        rodfilepersample = 2
        if rod_adc_bit == 8:
            rodfilepersample = 2
        self.rod_img_per_file = rodfilepersample
        self.rodfilepersample = rodfilepersample
        
        self.fileDict = dict([])
        self.fileList = []
        
        for DataTop in pathList:
            self.addMoreSample(self.fileDict,DataTop,MAXLEN,matchkey)
            
        self.sampleNumDict = dict([])
        self.sampleNum = 0
        
        for key in self.fileDict:
            self.sampleNumDict[key] =  self.dataNum(key)
            self.sampleNum += self.dataNum(key)
            print(key,'---legal sample num:',self.dataNum(key))
        
    def dataNum(self,key):
        return len(self.fileDict[key]['trainData'])
        
    def __len__(self):
        return self.sampleNum 
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        for key in self.sampleNumDict:
            fmt_str = key + ' sample cell num:' + self.sampleNumDict[key] + '\n'
        return fmt_str
    
    def readRodFast(self,key,rod_id):
        if rod_id >= len(self.fileDict[key]['rod'])*self.rod_img_per_file:
            print('invalid coneid for ',len(self.fileDict[key]['rod']),' rod data')
            return None,None,-1
        
        rodFileID = rod_id // self.rod_img_per_file
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
            
        rodtimeStamp += (rod_id % self.rod_img_per_file)*131
            
        return sd_list[rod_id % self.rod_img_per_file],td_list[rod_id % self.rod_img_per_file],rodtimeStamp
            
    def readConeFast(self,key,cone_id,viz=True,useISP=False,ifSync =True):
        if cone_id >= len(self.fileDict[key]['cone']):
            print('invalid coneid for ',len(self.fileDict[key]['cone']),' cone data')
            return None,-1
        
        conefilename = self.fileDict[key]['cone'][cone_id]
        conetimeStamp = findDataTimestamp(conefilename)
        size = os.path.getsize(conefilename)
        #wihei = int(np.sqrt((size - 64) // 4))
        raw_vec = np.zeros(self.cone_height * self.cone_width, dtype=np.int16)
        rdc.cone_reader_py_byfile(conefilename, size // 4, raw_vec, self.cone_height, self.cone_width)
        cone_raw = np.reshape(raw_vec, (self.cone_height, self.cone_width))
        start_frame = cone_raw
        #rgb_processed = start_frame
        return start_frame, conetimeStamp

    def packRead(self,idx,key,conevsrodRate = 26,ifSync =True, needPreProcess = True):
        sample = dict([])
        
        filePackage = self.fileDict[key]['trainData'][idx]
        
        coneRange = filePackage['cone']
        rodRange = filePackage['rod']
        
        start_frame,coneTimeStamp1 = self.readConeFast(key,coneRange[0])
        end_frame,coneTimeStamp2 = self.readConeFast(key,coneRange[1])

        itter = rodRange[1] - rodRange[0]
        if itter<0:
            print(key,coneStartId, cone_id, coneRange)
            print(itter , rodRange[1] , rodRange[0])
            
        tsd = torch.zeros([3,itter,160,160])
        for i in range(itter):
            sd,td,rodTimeStamp = self.readRodFast(key,rodRange[0] + i)
            sdl,sdr = sd
            tsd[0,i,:,:] = torch.Tensor(td)
            tsd[1,i,:,:] = torch.Tensor(sdl)
            tsd[2,i,:,:] = torch.Tensor(sdr)
            if i ==0:
                sample['rt1']= rodTimeStamp
            if i ==itter-1:
                sample['rt2']= rodTimeStamp

        if needPreProcess:
            start_frame,end_frame,tsd  = self.preprocess(start_frame,end_frame,tsd)

        sample['F0'] = start_frame
        sample['F1'] = end_frame
        sample['tsdiff'] = tsd
        sample['key'] = key
        sample['idx'] = idx
        sample['ct1'] = coneTimeStamp1
        sample['ct2'] = coneTimeStamp2

        return sample
    
    def locateSample(self,index):
        relativeIndex = index
        for key in self.sampleNumDict:
            numKey = self.sampleNumDict[key]
            if relativeIndex >= numKey:
                relativeIndex -= numKey
            else:
                return key,relativeIndex
            
    def preprocess(self,F0,F1,tsdiff):
        F0 = lyncam_raw_comp(F0)
        F0 = demosaicing_npy(F0, 'bggr', 1, 10)/1024.0
        F1 = lyncam_raw_comp(F1)
        F1 = demosaicing_npy(F1, 'bggr', 1, 10)/1024.0
        tsdiff = F.interpolate(tsdiff,(320,640),mode='bilinear')/128.0
        return F0,F1,tsdiff
    
    def __getitem__(self, index):
        key,relativeIndex = self.locateSample(index)
        sample = self.packRead(relativeIndex, key)
        return sample