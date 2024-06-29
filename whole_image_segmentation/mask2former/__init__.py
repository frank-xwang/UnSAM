# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.sam_instance_tsv_dataset_mapper import SamInstanceTSVDatasetMapper
from .data.dataset_mappers.sam_instance_tsv_self_train_dataset_mapper import SamSelfTrainTSVDatasetMapper

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.coco_evaluation import COCOEvaluator
