export DETECTRON2_DATASETS=/PATH/TO/YOUR/DATASETS
export TRAIN_DATASETS=/PATH/TO/YOUR/SA-1B
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --num-gpus 4 \
  --config-file configs/maskformer2_R50_bs16_50ep.yaml \
  SOLVER.IMS_PER_BATCH 4 \
  DATALOADER.NUM_WORKERS 1 \
  OUTPUT_DIR ex_train \