 #python export.py --weights ./weights/YOLOv5_775emix_320x160_fp32.pt --include onnx --batch-size 1 --device 7 --img-size 160 320
 
python export.py --weights ./weights/YOLOv5COCO_80e.pt --include onnx --batch-size 1 --device 0 --img-size 160 320
 