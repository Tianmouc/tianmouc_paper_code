#modify the config file

# _C.MODEL.IMAGE_SIZE = [320, 640]  # width * height, ex: 192 * 256
# _C.DATASET.DATAROOT = '/data/bdd100k/images'       # the path of images folder
# _C.DATASET.LABELROOT = '/data/bdd100k/labels'      # the path of det_annotations folder
# _C.DATASET.MASKROOT = '/data/bdd100k/bdd_seg_gt'                # the path of da_seg_annotations folder
# _C.DATASET.LANEROOT = '/data/bdd100k/bdd_lane_gt'               # the path of ll_seg_annotations folder

python ./tools/train.py