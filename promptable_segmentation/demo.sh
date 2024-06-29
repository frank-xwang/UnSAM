# #interactive gradio demo
CUDA_VISIBLE_DEVICES=4 python demo_promptable.py \
    --ckpt /home/xudongw/UnSAM-Semantic/data/output/gpu4-bs4-lr1e-4-iter100k-12jsons-SSL-AllAnnos-NMask120-Thresh0.02/10k_0099999.pth \
    --conf_files configs/semantic_sam_only_sa-1b_swinT.yaml \
    --device cpu