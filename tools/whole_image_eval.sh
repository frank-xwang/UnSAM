export DETECTRON2_DATASETS=/PATH/TO/YOUR/DATASETS
export TRAIN_DATASETS=/PATH/TO/YOUR/SA-1B
CUDA_VISIBLE_DEVICES=5,6,7,8, python whole_image_segmentation/train_net.py \
  --num-gpus 4 \
  --config-file whole_image_segmentation/configs/maskformer2_R50_bs16_50ep.yaml \
  --eval-only \
  MODEL.WEIGHTS /home/xudongw/mask2former/output/bs16_lr5e-5_rn50_41json_500masks_2000q_DINO/model_0199999.pth \
  SOLVER.IMS_PER_BATCH 4 \
  DATALOADER.NUM_WORKERS 1 \
  OUTPUT_DIR eval_output \