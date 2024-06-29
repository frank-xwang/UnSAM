import argparse
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

import json
import os
from tqdm import tqdm


def merge_dt_list(list):
    merged_data = {
        "info": {
            "year": 2023,
            "version": "1",
            "date_created": "no need record"
        },
        "images": [],
        "annotations": [],
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "hd",
                "supercategory": ""
            }
        ]
    }

    annotation_id_counter = 1

    image_ids = []

    for ann_path in tqdm(os.listdir(list)):
        if not ann_path.startswith('p'): continue

        ann = json.load(open(os.path.join(list, ann_path)))
        for data in ann["annotations"]:

            if data["image_id"] not in image_ids:
                image_ids.append(data["image_id"])
                merged_data["images"].append(
                    {
                        "id": data["image_id"],
                        "height": data["segmentation"]["size"][0],
                        "width": data["segmentation"]["size"][1],
                    }
                )

            ann_area = mask_util.area(data["segmentation"]).tolist()

            # one json only has one image in SA-1B
            data["id"] = annotation_id_counter
            annotation_id_counter += 1
            data["category_id"] = 1
            data["iscrowd"] = 0
            data["score"] = 1
            data["area"] = ann_area

            # Append the updated annotation to the merged_data
            merged_data["annotations"].append(data)

    # Save the merged data to a new JSON file
    output_path = "merged_dt.json"
    print(f"Saving merged data to {output_path}")
    with open(output_path, 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)
    return output_path

def merge_gt_files(coco_dt):
    merged_data = {
        "info": {
            "year": 2023,
            "version": "1",
            "date_created": "no need record"
        },
        "images": [],
        "annotations": [],
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "hd",
                "supercategory": ""
            }
        ]
    }

    annotation_id_counter = 1

    for dt_img in tqdm(coco_dt.dataset["images"]):
        name = "sa_"+str(dt_img["id"])+".json"
        f = open(os.path.join("datasets/sa1b/annotations/val_gt_2", name)) 
        data = json.load(f)
        if data == None: continue

        # one json only has one image in SA-1B
        if type(data["image"]) is not list:
            # SA only has image_id, not id
            data["image"]["id"] = data["image"]["image_id"]
            merged_data["images"].append(data["image"])

        # Update annotation IDs and image IDs
        for annotation in data["annotations"]:
            annotation["id"] = annotation_id_counter
            annotation_id_counter += 1
            annotation["score"] = 1.0
            annotation["category_id"] = 1
            annotation["iscrowd"] = 0

            # one json only has one image in SA-1B, and annotations don't have image_id
            if type(data["image"]) is not list:
                annotation["image_id"] = data["image"]["id"]
            
            # Append the updated annotation to the merged_data
            merged_data["annotations"].append(annotation)

    # Save the merged data to a new JSON file
    output_path = "merged_gt.json"
    with open(output_path, 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)
    return output_path



if __name__ == "__main__":

    parser = argparse.ArgumentParser('')
    parser.add_argument('--predict-directory', type=str, default="divide_and_conquer/pseudo_masks", help='predict-directory')
    parser.add_argument('--iou-type', type=str, default="segm", help='iou_type')

    args = parser.parse_args()

    path = merge_dt_list(args.predict_directory)

    coco_dt = COCO(path)
    coco_gt = COCO(merge_gt_files(coco_dt))

    # for ann in coco_dt.dataset["annotations"]:
    #     ann["score"] = 1

    coco_eval = COCOeval(coco_gt, coco_dt, args.iou_type)
    coco_eval.params.useCats = 0
    coco_eval.params.maxDets = [1, 100, 1000]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap = coco_eval.stats[:6]
    ar = coco_eval.stats[6:12]

    mAP_copypaste = (
        f'{ap[0]*100:.2f} {ap[1]*100:.2f} {ap[2]*100:.2f} {ap[3]*100:.2f} {ap[4]*100:.2f} {ap[5]*100:.2f}')
    mAR_copypaste = (
        f'{ar[0]*100:.2f} {ar[1]*100:.2f} {ar[3]*100:.2f} {ar[4]*100:.2f} {ar[5]*100:.2f} {ar[2]*100:.2f}')

    print("mAP copy-paste: ", mAP_copypaste)
    print("mAR copy-paste: ", mAR_copypaste)
    print("All in one copy-paste: ", mAP_copypaste + " " + mAR_copypaste)
    print("num of masks: ", len(coco_dt.dataset['annotations']))