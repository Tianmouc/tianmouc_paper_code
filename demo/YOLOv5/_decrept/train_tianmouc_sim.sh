# python train.py --img 320 --batch 32 --epochs 100 --data bdd100k.yaml --cfg yolov5_tianmouc.yaml
 
conda activate yolo
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11345  train_tianmouc.py --img 320 --batch-size 256 --epochs 100 --data bdd100k.yaml --cfg yolov5_tianmouc.yaml --save-period 1 --sync-bn --label-smoothing 0.05 --project runs/train_tianmouc