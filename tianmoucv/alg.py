import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy import signal
from PIL import Image
from tianmoucv import *

from .basic import *


def white_balance(img):

    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
    Ba[Ba>255] = 255
    Ga[Ga>255] = 255
    Ra[Ra>255] = 255
    img[:, :, 0] = Ba
    img[:, :, 1] = Ga
    img[:, :, 2] = Ra
    return img

def gaussain_kernel(size=5,sigma=2):
    size = int(size)
    if size % 2 == 0:
        size = size + 1
    m = (size - 1) / 2
    y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-m, m + 1))
    kernel = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel

def gaussian_smooth(inputTensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding_size = kernel.shape[-1] // 2
    input_padded = F.pad(inputTensor, (padding_size, padding_size, padding_size, padding_size), mode='reflect')
    kernel = kernel.to(inputTensor.device)
    return F.conv2d(input_padded, kernel, stride=1, padding=0)



def HarrisCorner(Ix,Iy,k = 0.5,th = 0.95,size=5,sigma=2,nmsSize=11):
    # 1. get difference image
    Ix[Ix<torch.max(Ix)*0.1]=0
    Iy[Iy<torch.max(Iy)*0.1]=0
    # 2. sober filtering
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
   # 3. windowed
    kernel = gaussain_kernel(size,sigma)
    Ix2 = gaussian_smooth(Ix2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Iy2 = gaussian_smooth(Iy2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Ixy = gaussian_smooth(Ixy.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    # 4. corner detect
    out = np.zeros(Ix2.shape)
    R = (Ix2 * Iy2 - Ixy ** 2) - k * ((Ix2 + Iy2) ** 2)
    threshold =  float(torch.max(R)) * th
    
    R_Max = F.max_pool2d(R.unsqueeze(0).unsqueeze(0), kernel_size=nmsSize, 
                             stride=1, padding=nmsSize//2).squeeze(0).squeeze(0)
    idmap = (R >= threshold).int() * (R > R_Max-1e-5).int()
    R = R[idmap>0]
        
    return idmap,R
 

def TomasiCorner(Ix, Iy, index=1000,size=5,sigma=2,nmsSize=11):
    # 1. get difference image
    Ix[Ix<torch.max(Ix)*0.1]=0
    Iy[Iy<torch.max(Iy)*0.1]=0
    # 2. sober filtering
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    # 3. windowed
    kernel = gaussain_kernel(size,sigma)
    Ix2 = gaussian_smooth(Ix2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Iy2 = gaussian_smooth(Iy2.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    Ixy = gaussian_smooth(Ixy.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    # prepare output image
    out = np.zeros(Ix2.shape)
    # get R
    K = Ix2**2 + Iy2 **2 + Iy2*Ix2 + Ixy**2 + 1e-16
    R = Ix2 + Iy2 - torch.sqrt(K)
    # detect corner
    sorted_, _ = torch.sort(R.view(1,-1), descending=True)#descending为False，升序，为True，降序
    threshold = sorted_[0,index]
    R_Max = F.max_pool2d(R.unsqueeze(0).unsqueeze(0), kernel_size=nmsSize, 
                             stride=1, padding=nmsSize//2).squeeze(0).squeeze(0)
    idmap = (R >= threshold).int() * (R > R_Max-1e-5).int()
    return idmap,R


def sift(Ix,Iy, keypoints):
    Ix = Ix.numpy().astype(np.float)
    Iy = Iy.numpy().astype(np.float)
    radius = 5
    
    descriptors = []
    
    count = 0
    for kp in keypoints:
        descriptorlist = None
        x, y = int(kp[0]), int(kp[1])
        magnitude, majorAngle = cv2.cartToPolar(Ix[x,y], Iy[x,y], angleInDegrees=True)
        majorAngle = majorAngle[0]
        pIx = Ix[y - radius : y + radius, x - radius : x + radius]
        pIy = Iy[y - radius : y + radius, x - radius : x + radius]
        shapeofIxy = pIx.shape
        step = int(radius / 4)
        for i in range(4):
            for j in range(4):
                dx = pIx[i * step : (i + 1) * step, j * step : (j + 1) * step]
                dy = pIy[i * step : (i + 1) * step, j * step : (j + 1) * step]
                magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
                if angle is None:
                    continue
                hist, _ = np.histogram(angle-majorAngle, bins=8, range=(0, 360), weights=magnitude)
                if descriptorlist is None:
                    descriptorlist = torch.Tensor(hist)
                else:
                    descriptorlist.extend(torch.Tensor(hist))

        descriptors.append(torch.stack(descriptorlist,dim=0))
        count += 1
    #print(descriptors)
    return descriptors


def feature_matching(des1, des2, ratio=0.7):
    """Match SIFT descriptors between two images."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1.cpu().numpy(), des2.cpu().numpy(), k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])
    return good_matches


def align_images(image,siftList1, siftList2, kpList1, kpList2, canvas=None):
    matches = feature_matching(siftList1, siftList2, ratio=0.7)
    H = None
    src_pts = []
    dst_pts = []
    if(len(matches)>4):
        src_pts = np.float32([kpList1[m[0].queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpList2[m[0].trainIdx] for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(H)
    for i in range(len(src_pts)):
        y1 , x1 = int(src_pts[i][0][0]),int(src_pts[i][0][1])
        y2 , x2 = int(dst_pts[i][0][0]),int(dst_pts[i][0][1])
        if canvas is not None:
            cv2.line(canvas,(x1,y1),(x2+640,y2),(255,0,0))
        print(x1,',',y1,'---',x2,',',y2)
        
    w,h = image.shape[1],image.shape[0]
    
    imagewp = image
    if H is not None:
        imagewp = cv2.warpPerspective(image,H, (w,h))
    return imagewp,H


def HarrisCorner3(Ix,Iy,It,k = 0.5,th = 0.95,size=5,sigma=2):
    # 1. get difference image
    Ix[Ix<torch.max(Ix)*0.1]=0
    Iy[Iy<torch.max(Iy)*0.1]=0
    Iy[It<torch.max(It)*0.1]=0
    grad_norm = (Ix**2 + Iy**2+ It**2)**0.5 + 1e-9
    Ix = Ix / torch.max(grad_norm)
    Iy = Iy / torch.max(grad_norm)
    It = It / torch.max(grad_norm)
    
    # 3. sober filtering
    A = Ix * Ix
    B = Iy * Iy
    C = It * It
    D = Ix * Iy
    E = Ix * It
    F = Iy * It
    
    size = int(size)
    if size % 2 == 0:
        size = size + 1
    m = (size - 1) / 2
    y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-m, m + 1))
    kernel = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    A = gaussian_smooth(A.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    B = gaussian_smooth(B.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    C = gaussian_smooth(C.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    D = gaussian_smooth(D.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    E = gaussian_smooth(E.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    F = gaussian_smooth(F.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    
    # 4. corner detect
    detM = A*B*C+2*D*E*F
    traceM = A+B+C-A*D*D-B*E*E-C*F*F
    R = detM - k * traceM ** 2
    threshold =  float(torch.max(R)) * th
    idmap = R >= threshold
    return idmap

class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """
    def __init__(self, W, H, device):
        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        MAGIC_NUM =  0.5
        # Extract horizontal and vertical flows.
        self.W = flow.size(3)
        self.H = flow.size(2)
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u + MAGIC_NUM
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v + MAGIC_NUM
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut

def local_norm(SD):
    grad_norm = (SD[0,...]**2 + SD[1,...]**2 + 1e-18)**0.5 + 1e-9
    return SD / torch.max(grad_norm)
        
def cal_optical_flow(SD,TD,win=5,stride=0,mask=None,ifInterploted = False):
    '''
    [dx,dy]*[dI/dx,DI/dy] + DI/dt = 0
    '''
    I = SD.size(-2)
    J = SD.size(-1)
    i_step  = win//2
    j_step  = win//2
    if stride == 0:
        stride =  win//2
    flow = torch.zeros([2,I//stride,J//stride])
    
    #加权
    Ix = SD[0,...]
    Iy = SD[1,...]
    It = TD[0,...]

    musk = torch.abs(It)>4
    It *= musk
    for i in range(i_step,I-i_step-1,stride):
        for j in range(j_step,J-j_step-1,stride):
            dxdy = [0,0]
            #忽略一些边界不稠密的点
            if mask is not None and np.sum(mask[i-i_step:i+i_step+1,j-j_step:j+j_step+1]) < 5:
                continue
            #取一个小窗口
            Ix_win = Ix[i-i_step:i+i_step+1,j-j_step:j+j_step+1].reshape(1,-1)
            Iy_win = Iy[i-i_step:i+i_step+1,j-j_step:j+j_step+1].reshape(1,-1)
            It_win = It[i-i_step:i+i_step+1,j-j_step:j+j_step+1].reshape(1,-1)            
            A = torch.cat([Ix_win,Iy_win],dim=0).transpose(1,0)
            B = -1 * It_win.reshape(1,-1).transpose(1,0)
            AT_B = torch.matmul(A.transpose(1,0),B)
            
            if torch.sum(AT_B) == 0:
                flow[0,i//stride,j//stride] = 0
                flow[1,i//stride,j//stride] = 0
                continue
            
            AT_A = torch.matmul(A.transpose(1,0),A)
            
            try :
                dxdy = np.linalg.solve(AT_A.cpu().numpy(), AT_B.cpu().numpy())
                flow[0,i//stride,j//stride] = float(dxdy[0])
                flow[1,i//stride,j//stride] = float(dxdy[1])
            except Exception as e :
                pass
    if not ifInterploted:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J*2), mode='bilinear').squeeze(0)
    else:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J), mode='bilinear').squeeze(0)
    return flow


def recurrentOF(SD,TD,ifInterploted = False):
    '''
    Ref: https://kns.cnki.net/kcms2/article/abstract?v=3uoqIhG8C475KOm_zrgu4h_jQYuCnj_co8vp4jCXSivDpWurecxFtEV8HAD0GySfgFWAxYnv5c-oQfA7zWjworscSCTy1fWb&uniplatform=NZKPT

    '''
    epsilon = 1e-8
    maxIteration = 50
    
    def uitter(u,v,Ix,Iy,It,lambdaL):
        newu = u - Ix * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newu
    def vitter(u,v,Ix,Iy,It,lambdaL):
        newv = v - Iy * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newv
        
    uitter_vector = np.vectorize(uitter)
    vitter_vector = np.vectorize(vitter)
        
    I = SD.size(-2)
    J = SD.size(-1)
    
    #加权
    Ix = SD[0,...].numpy()
    Iy = SD[1,...].numpy()
    It = TD[0,...].numpy()
    
    u = np.zeros([I,J])
    v = np.zeros([I,J])

    lambdaL = np.ones([I,J])
    
    for it in range(maxIteration):
        u_new = uitter_vector(u,v,Ix,Iy,It,lambdaL)
        v_new = vitter_vector(u,v,Ix,Iy,It,lambdaL)
        erroru = abs(u_new-u)
        errorv = abs(v_new-v)
        u = u_new
        v = v_new
        if np.max(erroru) < epsilon and np.max(errorv) < epsilon:
            break
    flow = torch.stack([torch.FloatTensor(u),torch.FloatTensor(v)],dim=0)
    
    if not ifInterploted:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J*2), mode='bilinear').squeeze(0)
    else:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J), mode='bilinear').squeeze(0)
    return flow


def recurrentMultiScaleOF(SD,TD,ifInterploted = False):
    import cv2

    epsilon = 1e-8
    maxIteration = 50
    scales = 4
    ld = 5

    def uitter(u,v,Ix,Iy,It,lambdaL):
        newu = u - Ix * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newu
    def vitter(u,v,Ix,Iy,It,lambdaL):
        newv = v - Iy * (Ix*u + Iy * v + It) / (lambdaL*lambdaL + Ix*Ix + Iy*Iy)
        return newv
        
    uitter_vector = np.vectorize(uitter)
    vitter_vector = np.vectorize(vitter)
        
    I = SD.size(-2)
    J = SD.size(-1)
    
    #加权
    Ix = SD[0,...].numpy()
    Iy = SD[1,...].numpy()
    It = TD[0,...].numpy()
    musk = abs(It)>4
    It *= musk
    
    factor = 2**(scales-1)
    u = np.zeros([I,J])
    v = np.zeros([I,J])
    for s in range(scales):
        factor = 2**(scales-s-1)
        lambdaL = np.ones([I//factor,J//factor]) * ld
        
        u =  cv2.resize(u, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        v =  cv2.resize(v, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        Ixs =  cv2.resize(Ix, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        Iys =  cv2.resize(Iy, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        Its =  cv2.resize(It, [I//factor,J//factor], interpolation=cv2.INTER_LINEAR)
        continueFlag = False
        for it in range(maxIteration):
            if continueFlag:
                continue
            u_new = uitter_vector(u,v,Ixs,Iys,Its,lambdaL)
            v_new = vitter_vector(u,v,Ixs,Iys,Its,lambdaL)
            erroru = abs(u_new-u)
            errorv = abs(v_new-v)
            u = u_new
            v = v_new
            if np.max(erroru) < epsilon and np.max(errorv) < epsilon:
                continueFlag = True
    
    flow = torch.stack([torch.FloatTensor(u),torch.FloatTensor(v)],dim=0)
        
    if not ifInterploted:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J*2), mode='bilinear').squeeze(0)
    else:
        flow = F.interpolate(flow.unsqueeze(0), size=(I,J), mode='bilinear').squeeze(0)
    return flow


class opticalDetector_Maxone():
    
    def __init__(self,noiseThresh=8,distanceThresh=0.2):
        self.noiseThresh = noiseThresh
        self.th = distanceThresh
        self.accumU = 0
        self.accumV = 0
        
    def __call__(self,sd,td,ifInterploted = False):
        
        td[abs(td)<self.noiseThresh] = 0
        sd[abs(sd)<self.noiseThresh] = 0
       
        #rawflow = cal_optical_flow(sd,td,win=7,stride=3,mask=None,ifInterploted = ifInterploted)
        #rawflow = recurrentOF(sd,td,ifInterploted = ifInterploted)
        rawflow = recurrentMultiScaleOF(sd,td,ifInterploted = ifInterploted)
        
        flow = flow_to_image(rawflow.permute(1,2,0).numpy())
        
        flowup = np.zeros([flow.shape[0]*2,flow.shape[1]*2,3])
        flowup[1::2,1::2,:] = flow/255.0
        flowup[0::2,1::2,:] = flow/255.0
        flowup[1::2,0::2,:] = flow/255.0
        flowup[0::2,0::2,:] = flow/255.0

        u = rawflow.permute(1,2,0).numpy()[:, :, 0]
        v = rawflow.permute(1,2,0).numpy()[:, :, 1]
        uv = [u,v]

        distance = ((u)**2 + (v)**2) *(u<0)
        
        distance[distance>self.th] = 1
        distance[distance<self.th] = 0
        distanceup = np.zeros([flow.shape[0]*2,flow.shape[1]*2])

        kernel = np.ones((3,3),np.uint8)              
        distance = cv2.dilate(distance,kernel,iterations=3) 

        distanceup[1::2,1::2] = distance * 255.0
        distanceup[0::2,1::2] = distance * 255.0
        distanceup[1::2,0::2] = distance * 255.0
        distanceup[0::2,0::2] = distance * 255.0
        f = (distanceup).copy().astype(np.uint8)
        contours,hierarchy = cv2.findContours(f,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(f, contours, -1, (0, 255, 255), 2)
        area = []
        box = None
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        if len(area)>0:
            if np.max(area) < 1200:
                return None,distanceup,flowup
            max_idx = np.argmax(area)
            for i in range(max_idx - 1):
                cv2.fillConvexPoly(f, contours[max_idx - 1], 0)
            cv2.fillConvexPoly(f, contours[max_idx], 255)
            maxcon = contours[max_idx]
            x1 = np.min(maxcon[:,:,0])  
            x2 = np.max(maxcon[:,:,0])  
            y1 = np.min(maxcon[:,:,1])  
            y2 = np.max(maxcon[:,:,1])  
            box = [x1,y1,x2,y2]
            #print(u[y1//2:y2//2,x1//2:x2//2]>0)
            #print(u[y1//2:y2//2,x1//2:x2//2],v[y1//2:y2//2,x1//2:x2//2])

        return box,distanceup,flowup