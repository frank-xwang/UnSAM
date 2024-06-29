# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Note:
For your custom dataset, there is no need to hard-code metadata anywhere in the code.
For example, for COCO-format dataset, metadata will be obtained automatically
when calling `load_coco_json`. For other dataset, metadata may also be obtained in other ways
during loading.

However, we hard-coded metadata for a few common dataset here.
The only goal is to allow users who don't have these dataset to use pre-trained models.
Users don't have to download a COCO json (which contains metadata), in order to visualize a
COCO model (with correct class names and colors).
"""
SA1B_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "fg"},
]

def _get_sa1b_instances_meta():
    thing_ids = [k["id"] for k in SA1B_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SA1B_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SA1B_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "class_image_count":  [{'id': 1, 'image_count': 116986}]
    }
    return ret


def _get_builtin_metadata(dataset_name):
    return _get_sa1b_instances_meta()
