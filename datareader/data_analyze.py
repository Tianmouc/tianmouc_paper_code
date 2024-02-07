import numpy as np
import os, sys
import cv2
import matplotlib.pyplot as plt
import time
# sys.path.append("/home/nvidia/Desktop/dataAnalys/LyncamSim_V1.0")
import rawDataReadTool as rdrt
import matplotlib.pyplot as plt
#import torch
from lib.utils import lyncam_raw_comp, visualize_tdiff, visualize_sdiff, rm_r_ts_offset
# from lib.utils import 
from lib.basic_isp import demosaicing_npy,AWBeasy
from tqdm import tqdm
# parameter setting
# set1: 8b_757fps_flicker, img_per_file = 2, adc_bit_prec = 8, fps = 757, aimDataSet = '8b_757fps_flicker'
# set2: 2b_1000fps_Lightning, img_per_file = 8, adc_bit_prec = 2, fps = 10000, aimDataSet = '2b_1000fps_Lightning'
img_per_file = 2
adc_bit_prec = 8
fps = 757
rod_cone_ratio = 25 if adc_bit_prec == 8 else 330
dataset_top = '../data_submit/'#'../data_submit/'
aimDataSet = '8bit_757fps_flicker'#8bit_757fps_flicker'# '2bit_1000fps_Lightning'
# for 8bit_757fps_flicker, start = 0, length = 2
start = 0 # start point in the file list
length =  2 # stop point in the file list
mydata = rdrt.TianmoucDataRead(rod_adc_bit=adc_bit_prec,rodfilepersample=img_per_file,ext=".bin",
                               dataset_top = dataset_top)

raw_files = mydata.readFileSeq(key=aimDataSet, cone_start=start,cone_duration=length, rod_cone_ratio=rod_cone_ratio)

viz = True
save_video = False
save_pic = False

if save_pic:
# create bmp save directory
    bmp_save_dir = os.path.join(os.path.join(dataset_top, aimDataSet), "diff_bmp")
    td_save_dir = os.path.join(bmp_save_dir, "td")
    sd_save_dir = os.path.join(bmp_save_dir, "sd")


    if os.path.exists(bmp_save_dir) is False:
        os.mkdir(bmp_save_dir)
    if os.path.exists(td_save_dir) is False:
        os.mkdir(td_save_dir)
    if os.path.exists(sd_save_dir) is False:
        os.mkdir(sd_save_dir)
if save_video:
    video_write_path = os.path.join(dataset_top, "{}_tdsd_pkt.avi".format(aimDataSet))

#dataMet = np.zeros([dataAmount])
#pkt_buf = np.empty(0, dtype=int)

if viz:
    save_size = (640 * 2, 320 * 2)
    fig, axes = plt.subplots(2, 2, figsize=(save_size[0] / 100, save_size[1] / 100))
    axes = axes.reshape(-1)
    plt.ion()
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

pkt_size_list = np.empty(0, dtype=int)
pkt_td_size_list = np.empty(0, dtype=int)
pkt_sd_size_list = np.empty(0, dtype=int)
td_ave_list = np.empty(0, dtype=float)
td_entp_list = np.empty(0, dtype=float)
sd_entp_list = np.empty(0, dtype=float)
rd_timestamp_list = np.empty(0, dtype=np.float32)

#start = 940
#stop =  start + 70 #dataAmount #dataAmount
num = 0
sd_ave = np.zeros([160, 320, 3], dtype=np.float64)
td_ave = np.zeros([160, 320, 3], dtype=np.float64)


if save_video:
    print('Saving to video: ' + video_write_path)
    out = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc(*'DIVX'), 4, save_size)


r_read_cnt = 0
c_read_cnt = 0
sd_vis = None
td_vis = None
for data_list in tqdm(raw_files):
    if data_list[0] == rdrt.CONE_TYPE:
        c_read_cnt += 1
        #print("C read cnt {}".format(c_read_cnt))
        c_cnt, c_ts, c_raw = data_list[1], data_list[2], data_list[3]
        raw_comp = lyncam_raw_comp(c_raw)

        rgb_vis = demosaicing_npy(raw_comp, 'bggr', 1, 10)
        rgb_vis = rgb_vis / 4
        rgb_vis = AWBeasy(rgb_vis)

        #rgb_vis = rgb_vis.astype(np.uint8)
#for i in range(0, 16):

    elif data_list[0] == rdrt.ROD_TYPE:
        r_cnt, r_ts, r_td, r_sdl, r_sdr = data_list[1], data_list[2], data_list[4], data_list[5], data_list[6]
        r_pkt_tot, r_pkt_sd, r_pkt_td = data_list[7], data_list[8], data_list[9]
        if r_read_cnt == 0:
            time_stamp_init = r_ts
        if adc_bit_prec != 2:
            r_pkt_tot *= 24
            r_pkt_sd *= 24
            r_pkt_td *= 24
        else:
            r_pkt_tot *= 32
            r_pkt_sd *= 32
            r_pkt_td *= 32
        pkt_size_list = np.append(pkt_size_list, r_pkt_tot)
        pkt_td_size_list = np.append(pkt_td_size_list, r_pkt_td)
        pkt_sd_size_list = np.append(pkt_sd_size_list, r_pkt_sd)
        rd_timestamp_list = np.append(rd_timestamp_list, rm_r_ts_offset(r_ts, time_stamp_init)) # uint: ms
        td_vis = visualize_tdiff(r_td, 8)
        sd_vis = visualize_sdiff(r_sdl, r_sdr, 8)
        r_read_cnt +=1

        r_pkt_td_bw = r_pkt_td* fps / (1024*1024*8)
        r_pkt_sd_bw = r_pkt_sd * fps / (1024 * 1024 * 8)
        r_pkt_tot_bw = r_pkt_tot * fps / (1024 * 1024 * 8)
        print("TD BW {} MB/s, SD BW {} MB/s, TOTAL {} MB/s".format(r_pkt_td_bw,r_pkt_sd_bw,r_pkt_tot_bw))

    if sd_vis is not None:
        vis_list = [rgb_vis, sd_vis, td_vis, None]
    else:
        vis_list = [rgb_vis, None, None, None]
    #print(i)
    for k in range(len(vis_list)):
        if vis_list[k] is not None:
            axes[k].clear()
            axes[k].imshow(vis_list[k], cmap='gray')
    ax_stat = axes[3]
    ax_stat.xaxis.set_visible(True)
    ax_stat.yaxis.set_visible(True)
    ax_stat.clear()
    ax_stat.plot(np.arange(len(pkt_size_list)), pkt_size_list * fps / (1024*1024*8), color=[1, 0, 0], marker='.',
                label='GAER')
    ax_stat.plot(np.arange(len(pkt_td_size_list)), pkt_td_size_list * fps / (1024*1024*8), color=[0, 1, 0], marker='.')
    ax_stat.plot(np.arange(len(pkt_sd_size_list)), pkt_sd_size_list * fps / (1024*1024*8), color=[0, 0, 1], marker='.',
                label='GAER')
        # for pos and neg([img],

    fig.canvas.draw()
    #print("num is {} ts is {}".format(num, rod_timestamp))
    num += 1

    if save_pic:
        cv2.imwrite("{}/{}_{}.bmp".format(td_save_dir, num, r_ts), td_vis[:, :, [2, 1, 0]])
        cv2.imwrite("{}/{}_{}.bmp".format(sd_save_dir, num, r_ts), sd_vis[:, :, [2, 1, 0]])
    if save_video:
            save_fig = np.asarray(fig.canvas.buffer_rgba())[..., -2::-1]  # convert from RGBA to BGR
            out.write(save_fig)

    if viz:
        plt.draw()
        plt.pause(0.0000001)
bw_mean = np.mean(pkt_size_list) * fps / (1024*1024*8)
print(f"Bandwidth mean {bw_mean} MB/s")

# np.save(os.path.join(dataset_top, "{}_pktsize".format(aimDataSet)), pkt_size_list)
# np.save(os.path.join(dataset_top, "{}_pktsize_td".format(aimDataSet)), pkt_td_size_list)
# np.save(os.path.join(dataset_top, "{}_pktsize_sd".format(aimDataSet)), pkt_sd_size_list)
# scipy.io.savemat(os.path.join(dataset_top, "{}_pktsize_all.mat".format(aimDataSet)), \
#                  dict(pa=pkt_size_list, ptd = pkt_td_size_list, psd =pkt_sd_size_list, fps=fps))
