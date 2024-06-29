# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances


_PREDEFINED_SPLITS_UNSAM_SA1B = {}
_PREDEFINED_SPLITS_UNSAM_SA1B["unsam_sa1b"] = {
    "unsam_sa1b_val": ("sa1b/images", "sa1b/annotations/sa1b_val.json"),
}

def register_all_unsam_sa1b(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UNSAM_SA1B.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_UNSAM_ADE20K = {}
_PREDEFINED_SPLITS_UNSAM_ADE20K["unsam_ade20k"] = {
    "unsam_ade20k_val": ("ade/images", "ade/annotations/ade_val.json"),
}

def register_all_unsam_ade20k(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UNSAM_ADE20K.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_UNSAM_ENTITY = {}
_PREDEFINED_SPLITS_UNSAM_ENTITY["unsam_entity"] = {
    "unsam_entity_val": ("entity/images", "entity/annotations/entityseg_val.json"),
}

def register_all_unsam_entity(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UNSAM_ENTITY.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_UNSAM_PACO = {}
_PREDEFINED_SPLITS_UNSAM_PACO["unsam_paco"] = {
    "unsam_paco_val": ("paco/images", "paco/annotations/paco_val.json"),
}

def register_all_unsam_paco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UNSAM_PACO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_UNSAM_PARTIMAGENET = {}
_PREDEFINED_SPLITS_UNSAM_PARTIMAGENET ["unsam_partimagenet"] = {
    "unsam_partimagenet_val": ("partimagenet/images", "partimagenet/annotations/partimagenet_val.json"),
}

def register_all_unsam_partimagenet(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UNSAM_PARTIMAGENET.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_UNSAM_COCO = {}
_PREDEFINED_SPLITS_UNSAM_COCO ["unsam_coco"] = {
    "unsam_coco_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
}

def register_all_unsam_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UNSAM_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_UNSAM_LVIS = {}
_PREDEFINED_SPLITS_UNSAM_LVIS ["unsam_lvis"] = {
    "unsam_lvis_val": ("lvis/images", "lvis/annotations/lvis_v1_val.json"),
}

def register_all_unsam_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UNSAM_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_unsam_sa1b(_root)
    register_all_unsam_ade20k(_root)
    register_all_unsam_paco(_root)
    register_all_unsam_entity(_root)
    register_all_unsam_partimagenet(_root)
    register_all_unsam_coco(_root)
    register_all_unsam_lvis(_root)