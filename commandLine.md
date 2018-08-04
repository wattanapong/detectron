CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_e2e_mask_rcnn_R-101-FPN_1x_coco2014.txt

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014_2017.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_test_e2e_mask_rcnn_R-101-FPN_1x_coco2014valminusmini_coco2017.txt
	
CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014_train_valminusmini.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_test_e2e_mask_rcnn_R-101-FPN_1x_coco2014_train_valminusmini.txt
	
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_2x_coco2014_train_valminusmini_1gpu.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_test_e2e_mask_rcnn_R-101-FPN_2x_coco2014_train_valminusmini.txt
	
python tools/train_net.py \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_2x_coco2014_train_valminusmini_2gpu.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_test_e2e_mask_rcnn_R-101-FPN_2x_coco2014_train_valminusmini.txt
	
python tools/train_net.py \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_3x_coco2014_train_valminusmini_2gpu.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_test_e2e_mask_rcnn_R-101-FPN_2x_coco2014_train_valminusmini.txt
-----------------------------------------------------------------------
test
-----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1 python tools/test_net.py  --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014_2017.yaml \
    --multi-gpu-testing  \
    TEST.WEIGHTS /tmp/detectron-output/train/coco_2014_train:coco_2014_valminusminival:coco_2017_train/generalized_rcnn/model_final.pkl \
NUM_GPUS 2 > test_coco_2017_model_train_MaskRCNN_coco2017.txt

python tools/test_net.py  --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014_2017.yaml \
    TEST.WEIGHTS /tmp/detectron-output/train/coco_2014_valminusminival:coco_2017_train/generalized_rcnn/model_final.pkl \
NUM_GPUS 2 > test_coco_2014minival_model_train_MaskRCNN_coco2014mini_coco2017.txt

python tools/test_net.py  --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014_train_valminusmini_1gpu.yaml \
    TEST.WEIGHTS /tmp/detectron-output/train/coco_2014_valminusminival:coco_2014_train/generalized_rcnn/model_final.pkl \
NUM_GPUS 2 > test_coco_2014minival_model_train_MaskRCNN_coco2014.txt

python tools/test_net.py  --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_2x_coco2014_train_valminusmini_2gpu.yaml \
    TEST.WEIGHTS /tmp/detectron-output/train/coco_2014_valminusminival:coco_2014_train/generalized_rcnn/model_final.pkl \
NUM_GPUS 2 > test_coco_2014minival_model_train_MaskRCNN_coco2014.txt
-------------------------------------------------------------------------	
CUDA_VISIBLE_DEVICES=0,1 python tools/reval.py \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014_2017.yaml \
	--matlab False
-------------------------------------------------------------------------
copy -a
cp -a /tmp/detectron-output/train/coco_2014_valminusminival:coco_2014_train /home/wattanapongsu/library/detectron/detectron/weights/train/coco_2014_valminusminival:coco_2014_train
	
	
	