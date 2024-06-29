from semantic_sam.body.encoder.registry import model_entrypoints
from semantic_sam.body.encoder.registry import is_model

from semantic_sam.body.encoder.transformer_encoder_fpn import *
from .encoder_deform import *

def build_encoder(config, *args, **kwargs):
    model_name = config['MODEL']['ENCODER']['NAME']

    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)