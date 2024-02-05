import sys
sys.path.append("./datareader")
import torch.nn.functional as F
from .isp import lyncam_raw_comp,demosaicing_npy
import torch

def interpolate_preprocess(dataset,start,key,conevsrodRate = 25):
        itter = 26
        sample = dict([])
        rgb_processed,_,_,_,old_conetimeStamp,_ = dataset.readFile(key,start,0,viz=False,ifSync =True)
        rgb_processed = lyncam_raw_comp(rgb_processed)
        rgb = demosaicing_npy(rgb_processed, 'bggr', 1, 10)/1024
        sample['F0'] = torch.FloatTensor(rgb).permute(2,0,1)
        rgb_processed,_,_,_,new_conetimeStamp,_ = dataset.readFile(key,start+1,0,viz=False,ifSync =True)
        rgb_processed = lyncam_raw_comp(rgb_processed)
        rgb = demosaicing_npy(rgb_processed, 'bggr', 1, 10)/1024
        sample['F1'] = torch.FloatTensor(rgb).permute(2,0,1)
        tsd = torch.zeros([3,itter,320,640])
        gap = 0
        mingap = 0
        maxgap = 0
        bias = 0
        _,tdt,sdt,rodtimeStamp,_,_ = dataset.readFile(key,start,start*conevsrodRate+0+bias,viz=False,ifSync =True)
        mingap = abs(old_conetimeStamp-rodtimeStamp) 
        _,tdt,sdt,rodtimeStamp,_,_ = dataset.readFile(key,start,start*conevsrodRate+itter-1+bias,viz=False,ifSync =True)
        maxgap = abs(new_conetimeStamp-rodtimeStamp)
        if maxgap > 30 or mingap>30:
            bias += 1
        for i in range(itter):
            #时间对齐
            _,tdt,sdt,rodtimeStamp,_,_ = dataset.readFile(key,start,start*conevsrodRate+i+bias,viz=False,ifSync =True)
            if i == 0:
                mingap = abs(old_conetimeStamp-rodtimeStamp) 
            if i == itter-1:
                maxgap = abs(new_conetimeStamp-rodtimeStamp) 
            sdt = sdt.permute(2,0,1)
            td_inter = F.interpolate(tdt.unsqueeze(0).unsqueeze(0), 
                                                       (tdt.shape[0]*2,tdt.shape[1]*4), mode='bilinear')
            sd_inter = F.interpolate(sdt.unsqueeze(0), (sdt.shape[1]*2,sdt.shape[2]*4), mode='bilinear')
            td_inter = td_inter.squeeze(0).squeeze(0)
            sd_inter = sd_inter.squeeze(0)
            tsd[0,i,:,:] = td_inter
            tsd[1:3,i,:,:] = sd_inter

        sample['tsdiff'] = tsd
        return sample
    
    
def warp(sample,ReconModel, warper=None,h=320,w=640,device=torch.device('cuda:0'),ifsingleDirection=False):
    F0 = sample['F0'].unsqueeze(0)
    F1 = sample['F1'].unsqueeze(0)
    tsdiff = sample['tsdiff']/128
    tsdiff = tsdiff.unsqueeze(0)
    
    biasw = (640-w)//2
    biash = (320-h)//2
    
    batchSize = 50
    if ifsingleDirection:
        batchSize = 25
    Ft_batch = torch.zeros([batchSize,3,h,w])
    F_batch = torch.zeros([batchSize,3,h,w])
    SD0_batch = torch.zeros([batchSize,2,h,w])
    SD1_batch = torch.zeros([batchSize,2,h,w])
    td_batch = torch.zeros([batchSize,1,h,w])
    td_batch_inverse = torch.zeros([25,1,h,w])
    
    tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]

    for t in range(25):#F0->F1-dt
        SD0_batch[t,...] = tsdiff[:,1:,0,...]
        SD1_batch[t,...] = tsdiff[:,1:,t,...]
        F_batch[t,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
        if t == 0:
            td_batch[t,...] = 0
        else:
            td_batch[t,...] = torch.sum(tsdiff[:,0:1,1:t,...],dim=2)     
           
    if not ifsingleDirection:
        for t in range(25):#F0->F1-dt
            SD0_batch[t+25,...] = tsdiff[:,1:,25,...]
            SD1_batch[t+25,...] = tsdiff[:,1:,t,...]
            F_batch[t+25,...] = F1[:,:,biash:h+biash,biasw:w+biasw]
            if t == 24:
                td_batch[t+25,...] = 0
            else:
                td_batch[t+25,...] = torch.sum(tsdiff[:,0:1,t+1:,...],dim=2) * -1   
    Ft = None
    if not warper is None:
        Ft,_,_ = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                 SD0_batch.to(device), SD1_batch.to(device), warper)
    else:
        Ft,_,_,_ = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                 SD0_batch.to(device), SD1_batch.to(device))      
    
    for t in range(25):
        if ifsingleDirection:
            Ft_batch[t,...] = Ft[t,...]
        else:
            Ft_batch[t,...] = (Ft[t,...] + Ft[t+25,...])/2

    return Ft_batch,F_batch,tsdiff

    


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
    
    
#####################   
#sample input
#@F0:[1,3,w,h],0~1
#@F0:[1,3,w,h],0~1
#@F0:[1,3,w,h],-1~1
#####################
def warp_fast(sample,ReconModel, warper=None,h=320,w=640,device=torch.device('cuda:0'),ifsingleDirection=False):  
    F0 = sample['F0']
    F1 = sample['F1']
    tsdiff = sample['tsdiff']
    biasw = (640-w)//2
    biash = (320-h)//2
    batchSize = 50
    if ifsingleDirection:
        batchSize = 25
    Ft_batch = torch.zeros([batchSize,3,h,w])
    F_batch = torch.zeros([batchSize,3,h,w])
    SD0_batch = torch.zeros([batchSize,2,h,w])
    SD1_batch = torch.zeros([batchSize,2,h,w])
    td_batch = torch.zeros([batchSize,1,h,w])
    td_batch_inverse = torch.zeros([25,1,h,w])
    
    tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]

    for t in range(25):#F0->F1-dt
        SD0_batch[t,...] = tsdiff[:,1:,0,...]
        SD1_batch[t,...] = tsdiff[:,1:,t,...]
        F_batch[t,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
        if t == 0:
            td_batch[t,...] = 0
        else:
            td_batch[t,...] = torch.sum(tsdiff[:,0:1,1:t,...],dim=2)     
           
    if not ifsingleDirection:
        for t in range(25):#F0->F1-dt
            SD0_batch[t+25,...] = tsdiff[:,1:,-1,...]
            SD1_batch[t+25,...] = tsdiff[:,1:,t,...]
            F_batch[t+25,...] = F1[:,:,biash:h+biash,biasw:w+biasw]
            if t == 24:
                td_batch[t+25,...] = 0
            else:
                td_batch[t+25,...] = torch.sum(tsdiff[:,0:1,t+1:,...],dim=2) * -1   
    Ft = None
    if not warper is None:
        Ft,_,_ = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                 SD0_batch.to(device), SD1_batch.to(device), warper)
    else:
        Ft,_,_,_ = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                 SD0_batch.to(device), SD1_batch.to(device))      
    
    for t in range(25):
        if ifsingleDirection:
            Ft_batch[t,...] = Ft[t,...]
        else:
            Ft_batch[t,...] = (Ft[t,...] + Ft[t+25,...])/2

    return Ft_batch,F_batch,tsdiff

#####################   
#sample input
#@F0:[1,3,w,h],0~1
#@F0:[1,3,w,h],0~1
#@F0:[1,3,w,h],-1~1
#####################
def warp_add_gray(sample,ReconModel, warper=None,h=320,w=640,device=torch.device('cuda:0'),ifsingleDirection=False):  
    F0 = sample['F0'].permute(0,3,1,2).float().to(device)
    F1 = sample['F1'].permute(0,3,1,2).float().to(device)
    tsdiff = sample['tsdiff'].float().to(device)
    grays = sample['grays'].float().to(device)
    
    biasw = (640-w)//2
    biash = (320-h)//2
    batchSize = 50
    if ifsingleDirection:
        batchSize = 25
    Ft_batch = torch.zeros([batchSize,3,h,w])
    F_batch = torch.zeros([batchSize,3,h,w])
    SD0_batch = torch.zeros([batchSize,2,h,w])
    SD1_batch = torch.zeros([batchSize,2,h,w])
    td_batch = torch.zeros([batchSize,1,h,w])
    td_batch_inverse = torch.zeros([25,1,h,w])
    
    tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]
    grays1 = grays[0,:,:,biash:h+biash,biasw:w+biasw].permute(1,0,2,3)
    grays0 = torch.zeros([batchSize,1,h,w]).to(device)
    
    print(grays1.shape,grays0.shape)

    for t in range(25):#F0->F1-dt
        SD0_batch[t,...] = tsdiff[:,1:,0,...]
        SD1_batch[t,...] = tsdiff[:,1:,t,...]
        F_batch[t,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
        grays0[t,...] = grays[0,:,0:1,biash:h+biash,biasw:w+biasw]
        if t == 0:
            td_batch[t,...] = 0
        else:
            td_batch[t,...] = torch.sum(tsdiff[:,0:1,1:t,...],dim=2)     
           
    if not ifsingleDirection:
        for t in range(25):#F0->F1-dt
            SD0_batch[t+25,...] = tsdiff[:,1:,25,...]
            SD1_batch[t+25,...] = tsdiff[:,1:,t,...]
            F_batch[t+25,...] = F1[:,:,biash:h+biash,biasw:w+biasw]
            grays0[t+25,...] = grays[0,:,0:1,biash:h+biash,biasw:w+biasw]
            grays1[t+25,...] = grays[0,:,:,biash:h+biash,biasw:w+biasw].permute(1,0,2,3)
            if t == 24:
                td_batch[t+25,...] = 0
            else:
                td_batch[t+25,...] = torch.sum(tsdiff[:,0:1,t+1:,...],dim=2) * -1   
    Ft = None
    if not warper is None:
        Ft,_,_ = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                 SD0_batch.to(device), SD1_batch.to(device), grays0, grays1)
    else:
        Ft,_,_,_ = ReconModel(F_batch.to(device) , td_batch.to(device), 
                                 SD0_batch.to(device), SD1_batch.to(device), grays0, grays1)      
    
    for t in range(25):
        if ifsingleDirection:
            Ft_batch[t,...] = Ft[t,...]
        else:
            Ft_batch[t,...] = (Ft[t,...] + Ft[t+25,...])/2

    return Ft_batch,F_batch,tsdiff
