import argparse
import torch
import os
import tqdm
import PIL.Image as Image
import numpy as np
import segmentation_refinement as refine
from tqdm import tqdm
from torchvision import transforms
import dino
import cv2

from coco_annotator import create_image_info, create_annotation_info, output, category_info
from iterative_merging import iterative_merge
from cascadepsp import postprocess
from detectron2.config import get_cfg
from engine.defaults import DefaultPredictor
from pycocotools import mask as mask_utils
from detectron2.utils.colormap import random_color

from divide_conquer import setup_cfg

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config-file",
        default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
    )
    # backbone args
    parser.add_argument("--patch-size", default=8, type=int)
    parser.add_argument("--feature-dim", default=768, type=int)
    parser.add_argument("--backbone-size", default='base', type=str)
    parser.add_argument("--backbone-url", default="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth", type=str)

    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default="pseudo_masks_output")
    parser.add_argument("--preprocess", default=None, type=bool)
    parser.add_argument("--postprocess", default=None, type=bool)
    # preprocess args
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--start-id", default=None, type=int)
    parser.add_argument("--end-id", default=None, type=int)
    parser.add_argument("--local-size", default=256, type=int)
    parser.add_argument("--kept-thresh", default=0.9)
    parser.add_argument("--NMS-iou", default=0.9)
    parser.add_argument("--NMS-step", default=5)
    parser.add_argument("--thetas", default=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1], type=list)
    # postprocess args
    parser.add_argument("--crop-ratio", default=2.0)
    parser.add_argument("--refine-scale", default=1)
    parser.add_argument("--refine-min-L", default=100)
    parser.add_argument("--refine-max-L", default=900)
    parser.add_argument("--iou-thresh", default=0.5)
    parser.add_argument("--min-area-thresh", default=0.0)
    parser.add_argument("--max-area-thresh", default=0.9)
    parser.add_argument("--cover-thresh", default=0.9)
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def NMS(pool, threshold, step):
    # score is the area percent
    sorted_masks = sorted(pool, key=lambda mask: area(mask), reverse=True)
    masks_kept_indices = list(range(len(pool)))

    for i in range(len(sorted_masks)):
        if i in masks_kept_indices:
            for j in range(i+1, min(len(sorted_masks), i+step)):
                if iou(sorted_masks[i], sorted_masks[j]) > threshold:
                    masks_kept_indices.remove(j) if j in masks_kept_indices else None

    return [sorted_masks[i] for i in masks_kept_indices]

def area(mask):
    return np.count_nonzero(mask) / mask.size

def iou(mask1, mask2):
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(mask1) + np.count_nonzero(mask2) - intersection
    if union == 0: return 0
    return intersection / union

def coverage(mask1, mask2):
    if np.count_nonzero(mask1) == 0: return 0
    return np.count_nonzero(np.logical_and(mask1, mask2)) / np.count_nonzero(mask1)

def resize_mask(bipartition_masked, I_size):
    # do preprocess the mask before put into the refiner
    bipartition_masked = Image.fromarray(np.uint8(bipartition_masked*255))
    bipartition_masked = np.asarray(bipartition_masked.resize(I_size))
    bipartition_masked = bipartition_masked.astype(np.uint8)
    upper = np.max(bipartition_masked)
    lower = np.min(bipartition_masked)
    thresh = upper / 2.0
    bipartition_masked[bipartition_masked > thresh] = upper
    bipartition_masked[bipartition_masked <= thresh] = lower

    return bipartition_masked

def smallest_square_containing_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if len(np.where(rows)[0]) == 0 or len(np.where(cols)[0]) == 0:
        return 0, 1, 0, 1

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin, ymax, xmin, xmax

ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def generate_feature_matrix(backbone, image, feat_dim, feat_num):
    if next(backbone.parameters()).device == torch.device('cpu'):
        tensor = ToTensor(image).unsqueeze(0)
        feat = backbone(tensor)[0]
    else:
        tensor = ToTensor(image).unsqueeze(0).half()
        tensor = tensor.cuda()
        feat = backbone(tensor)[0].cpu()
    feat_reshaped = feat.reshape(feat_dim, feat_num, feat_num)
    feat_reshaped = feat_reshaped.permute(1, 2, 0)
    return feat_reshaped

def vis_mask(input, mask, mask_color) :
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.5 + np.array(mask_color) * 0.5).astype(np.uint8)
    return Image.fromarray(rgb)

def save_image(I, pool, output):
    # the visualization strategy is small masks on top of large masks
    already_painted = np.zeros(np.array(I).shape[:2])
    input = I.copy()
    i = 0
    for mask in tqdm(pool):
        already_painted += mask.astype(np.uint8)
        overlap = (already_painted == 2)
        if np.sum(overlap) != 0:
            input = Image.fromarray(overlap[:, :, np.newaxis] * np.copy(I) + np.logical_not(overlap)[:, :, np.newaxis] * np.copy(input))
            already_painted -= overlap
        input = vis_mask(input, mask, random_color(rgb=True))
    input.save(output)

def main():
    args = get_parser().parse_args()
    refiner = refine.Refiner(device='cuda:0')

    # divide-and-conquer algorithm
    # load the CutLER model
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    # load DINO backbone
    backbone = dino.ViTFeat(args.backbone_url, args.feature_dim, args.backbone_size, 'k', args.patch_size)
    backbone.eval()
    if cfg.MODEL.DEVICE == 'cpu':
        assert not args.postprocess, "postprocess needs gpu"
        backbone.to('cpu')
    else:
        backbone.cuda().to(torch.float16)

    segmentation_id = 1

    # coco format annotator initialization
    divide_conquer_masks = []
    output["image"], output["annotations"] = {}, []

    # Image import
    image_path = os.path.join(args.input)
    image = cv2.imread(image_path)
    H, W = image.shape[:2]

    # Divide phase
    predictions = predictor(image)
    divide_masks_tensor = predictions["instances"].get("pred_masks")
    divide_masks = []
    for i in range(divide_masks_tensor.shape[0]):
        divide_masks.append(divide_masks_tensor[i,:,:].cpu().numpy())
    divide_conquer_masks.extend(divide_masks)

    # Conquer phase
    for divide_mask in divide_masks:
        conquer_masks = []
        # find the bounding box and resize the original images
        ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
        if (ymax-ymin) <= 0 or (xmax-xmin) <= 0: continue
        local_image = image[ymin:ymax, xmin:xmax]
        resized_local_image = Image.fromarray(local_image).resize([args.local_size, args.local_size])

        feature_matrix = generate_feature_matrix(backbone, resized_local_image, args.feature_dim, args.local_size//args.patch_size)
        merging_masks = iterative_merge(feature_matrix, args.thetas)
        
        for layer in merging_masks:
            if layer.shape[0] == 0: continue

            for i in range(layer.shape[0]):
                mask = layer[i, :, :]
                mask = resize_mask(mask, [xmax-xmin, ymax-ymin])
                mask = (mask > 0.5 * 255).astype(int)

                if coverage(mask, divide_mask[ymin:ymax, xmin:xmax]) <= args.kept_thresh: continue
                enlarged_mask = np.zeros_like(divide_mask)
                enlarged_mask[ymin:ymax, xmin:xmax] = mask
                conquer_masks.append(enlarged_mask)

        conquer_masks = NMS(conquer_masks, args.NMS_iou, args.NMS_step)
        divide_conquer_masks.extend(conquer_masks)

    for m in divide_conquer_masks:
        # create coco-style annotation info 
        annotation_info = create_annotation_info(
            segmentation_id, 0, category_info, m.astype(np.uint8), None)
        if annotation_info is not None:
            output["annotations"].append(annotation_info)
            segmentation_id += 1
    
    # postprocess CascadePSP
    if args.postprocess:    
        refined_annotations = postprocess(args, refiner, output, image)
        output["annotations"] = refined_annotations["annotations"]
    
    visualized_masks = []
    for mask_encoded in output["annotations"]:
        mask = mask_utils.decode(mask_encoded['segmentation'])
        visualized_masks.append(mask)
    sorted_masks = sorted(visualized_masks, key=lambda m: area(m), reverse=True)

    save_image(image, sorted_masks, args.output)

if __name__ == "__main__":
    main()