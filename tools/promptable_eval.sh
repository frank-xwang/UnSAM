export DETECTRON2_DATASETS=/PATH/TO/YOUR/DATASETS
CUDA_VISIBLE_DEVICES=4, python promptable_segmentation/train_net.py \
    --eval_only \
    --num-gpus 1 \
    --config-file promptable_segmentation/configs/semantic_sam_only_sa-1b_swinT.yaml \
    COCO.TEST.BATCH_SIZE_TOTAL=1 \
    MODEL.WEIGHTS=/home/xudongw/UnSAM-Semantic/data/output/gpu4-bs4-lr1e-4-iter100k-12jsons-SSL-AllAnnos-NMask120-Thresh0.02/10k_0099999.pth \
    OUTPUT_DIR=eval_output \