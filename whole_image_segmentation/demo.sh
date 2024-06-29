python demo_whole_image.py \
    --input whole_image_segmentation/examples/sa_628955.jpg \
    --output demo.jpg \
    --opts \
    MODEL.WEIGHTS /home/xudongw/mask2former/output/bs16_lr5e-5_rn50_41json_500masks_2000q_DINO/model_0199999.pth \
    MODEL.DEVICE cpu