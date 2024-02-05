__author__ = 'Y. Lin'
__authorEmail__ = '532109881@qq.com'
import torch
import numpy as np


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

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))
    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.uint8(img)


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img


#viz diff
def vizDiff(diff,thresh=12):
    rgb_diff = 0
    w,h = diff.shape
    rgb_diff = torch.ones([3,w,h]) * 255
    diff[abs(diff)<thresh] = 0
    rgb_diff[0,...][diff>0] = 0
    rgb_diff[1,...][diff>0] = diff[diff>0]
    rgb_diff[2,...][diff>0] = diff[diff>0]
    rgb_diff[0,...][diff<0] = -diff[diff<0]
    rgb_diff[1,...][diff<0] = 0
    rgb_diff[2,...][diff<0] = -diff[diff<0]
    return rgb_diff

#axis mapping
#input: [w,h,2]
#output: [w,h],[w,h]
#mapping ruls: tianmouc pixel pattern

def fourdirection2xy(sd):
    print("warning: 0711version, Ix may be wrong direction")
    Ix = np.zeros(sd.shape[:2])
    Iy = np.zeros(sd.shape[:2])
           
    sdul = sd[0::2,...,0]
    sdll = sd[1::2,...,0]
    sdur = sd[0::2,...,1]
    sdlr = sd[1::2,...,1]
    Ix[::2,...] = Ix[1::2,...]= (-(sdul + sdll)/1.414 + (sdur + sdlr)/1.414)/2
    Iy[1::2,...]= Iy[::2,...] = ((sdur - sdlr)/1.414 + (sdul - sdll)/1.414)/2
    return Ix,Iy


def images_to_video(frame_list,name,Val_size=(512,256),fps = 30):
    size = (Val_size[0], Val_size[1]) # 需要转为视频的图片的尺寸
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for frame in frame_list:
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()
    
def BGR2GRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    gray = gray.astype(np.uint8)
    return gray
    

# vectorized by Y. Lin
# Function to apply Poisson blending to two images
def poisson_blend(Ix,Iy,iteration=50):
    lap_blend = np.zeros(Ix.shape)

    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.copy()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[1:-1,2:] -  Iy[1:-1,1:-1] 
                    + Iy[2:,1:-1] -  Ix[1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[2:,1:-1] + lap_blend_old[0:-2,1:-1] 
                                 + lap_blend_old[1:-1,2:] + lap_blend_old[1:-1,0:-2])

        lap_blend[1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if np.sum(np.abs(lap_blend - lap_blend_old)) < 0.1:
            #print("converged")
            break
    # Return the blended image
    return lap_blend
