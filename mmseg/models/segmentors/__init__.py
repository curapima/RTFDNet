# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .seg_tta import SegTTAModel
from .encoder_decoder_mult import EncoderDecoder_mult

__all__ = [
    'BaseSegmentor','SegTTAModel','EncoderDecoder_mult'
]
