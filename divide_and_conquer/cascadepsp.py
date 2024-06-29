import json
import os
import cv2
import numpy as np
from pycocotools import mask as mask_util
from tqdm import tqdm

def area(mask):
    return np.count_nonzero(mask) / mask.size

def iou(mask1, mask2):
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(mask1) + np.count_nonzero(mask2) - intersection
    if union == 0: return 0
    return intersection / union

def postprocess(args, refiner, annotations, image):
    H, W = image.shape[:2]

    start_id = annotations["annotations"][0]['id']
    curr_id = 0
    refined_annotations = []

    for annotation in tqdm(annotations["annotations"]):
        mask = mask_util.decode(annotation['segmentation'])

        bbox = annotation['bbox']
        x1, y1, w, h = bbox
        x_center = x1 + w / 2
        y_center = y1 + h / 2

        longer_side = max(w, h)
        x1_resized = int(max(0, x_center - longer_side))
        y1_resized = int(max(0, y_center - longer_side))
        x2_resized = int(min(W, x_center + longer_side))
        y2_resized = int(min(H, y_center + longer_side))

        image_crop = image[y1_resized:y2_resized, x1_resized:x2_resized, :]
        mask_crop = mask[y1_resized:y2_resized, x1_resized:x2_resized]

        L = max(min(max(x2_resized-x1_resized, y2_resized-y1_resized) * args.refine_scale, args.refine_max_L), args.refine_min_L)
        refined_mask_crop = refiner.refine(image_crop, mask_crop * 255, fast=True, L=L)
        refined_mask_crop = (refined_mask_crop > 128).astype(np.uint8)

        refined_mask = np.zeros((H, W), dtype=np.uint8)
        refined_mask[y1_resized:y2_resized, x1_resized:x2_resized] = refined_mask_crop

        if area(refined_mask) < args.min_area_thresh or area(refined_mask) > args.max_area_thresh:
            continue
        if iou(mask, refined_mask) < args.iou_thresh:
            continue

        binary_mask_encoded = mask_util.encode(np.asfortranarray(refined_mask))
        binary_mask_encoded['counts'] = binary_mask_encoded['counts'].decode('ascii')

        annotation['segmentation'] = binary_mask_encoded
        annotation['bbox'] = mask_util.toBbox(binary_mask_encoded).tolist()
        annotation['area'] = mask_util.area(binary_mask_encoded).tolist()
        annotation['id'] = start_id + curr_id
        curr_id += 0

        refined_annotations.append(annotation)

    annotations["annotations"] = refined_annotations
    return annotations