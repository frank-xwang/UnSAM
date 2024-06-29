python divide_conquer.py \
    --input-dir /PATH/TO/DATASETS \
    --output-dir pseudo_masks \
    --start-id 50 \
    --end-id 100 \
    --preprocess True \
    --opts MODEL.WEIGHTS cutler_cascade_final.pth

python divide_conquer.py \
    --input-dir /PATH/TO/DATASETS \
    --output-dir pseudo_masks\
    --postprocess True \
    --opts MODEL.WEIGHTS cutler_cascade_final.pth