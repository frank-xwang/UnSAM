import argparse
import argparse
from detectron2.engine import DefaultPredictor, default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.utils.colormap import random_color
from mask2former import add_maskformer2_config
import cv2
import os
from tqdm import tqdm
import PIL.Image as Image
import numpy as np

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def area(mask):
    if mask.size == 0: return 0
    return np.count_nonzero(mask) / mask.size

def vis_mask(input, mask, mask_color) :
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.5 + np.array(mask_color) * 0.5).astype(np.uint8)
    return Image.fromarray(rgb)

def save_image(I, pool, output_path):
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
    input.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", type=str, help="path of input image")
    parser.add_argument("--output", type=str, help="path to save output image")
    parser.add_argument("--confidence_thresh", type=float, default=0.5, help="path to save output image")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    pred = DefaultPredictor(setup(args))
    inputs = cv2.imread(args.input)
    pred.input_format = "BGR"

    outputs = pred(inputs)['instances']
    masks = []
    for score, mask in zip(outputs.scores, outputs.pred_masks):
        if score < args.confidence_thresh: continue 
        masks.append(mask.cpu().numpy())
    sorted_masks = sorted(masks, key=lambda m: area(m), reverse=True)
    print(f"You have {len(sorted_masks)} masks for this image")

    save_image(inputs, sorted_masks, args.output)

if __name__ == "__main__":
    main()