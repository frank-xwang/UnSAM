export TRAIN_DATASETS=/scratch/one_month/2024_05/xudongw/SAM_4perc/gt_merged_threshold0.02

run_name="gpu4-bs4-lr1e-4-iter400k-42jsons-gt"
ngpus=4
CUDA_VISIBLE_DEVICES=0,1,2,3, python train_net.py \
    --num-gpus ${ngpus} \
    --resume \
    --config-file configs/semantic_sam_only_sa-1b_swinT.yaml \
        SOLVER.BASE_LR=1e-4 \
        COCO.TEST.BATCH_SIZE_TOTAL=${ngpus} \
        SAM.TEST.BATCH_SIZE_TOTAL=${ngpus} \
        SAM.TRAIN.BATCH_SIZE_TOTAL=${ngpus} \
        TEST.EVAL_PERIOD=400000 \
        OUTPUT_DIR=data/output/${run_name} \

