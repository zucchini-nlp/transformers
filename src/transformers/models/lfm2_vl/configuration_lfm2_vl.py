# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LFM2-VL model."""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Lfm2VlConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Lfm2VlForConditionalGeneration`]. It is used to instantiate an
    Lfm2Vl model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Lfm2-VL-1.6B.

    e.g. [LiquidAI/LFM2-VL-1.6B](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vision_config (`AutoConfig | dict`,  *optional*, defaults to `Siglip2ImageConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`AutoConfig | dict`, *optional*, defaults to `Lfm2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 396):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        projector_hidden_size (`int`, *optional*, defaults to 2560):
            The hidden size of the multimodal projector.
        projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.
        projector_use_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to use layernorm in the multimodal projector.
        downsample_factor (`int`, *optional*, defaults to 2):
            The downsample_factor factor of the vision backbone.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the word embeddings of the text backbone.
    """

    model_type = "lfm2_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 396
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 2560
    projector_bias: bool = True
    projector_use_layernorm: bool = True
    downsample_factor: int = 2
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip2_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip2_vision_model"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "lfm2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["lfm2"]()

        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        super().__post_init__(**kwargs)


__all__ = ["Lfm2VlConfig"]
