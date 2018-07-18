CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_e2e_mask_rcnn_R-101-FPN_1x_coco2014.txt

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/wattanapong/train_e2e_mask_rcnn_R-101-FPN_1x_coco2014_2017.yaml \
    OUTPUT_DIR /tmp/detectron-output > train_test_e2e_mask_rcnn_R-101-FPN_1x_coco2014_2017.txt