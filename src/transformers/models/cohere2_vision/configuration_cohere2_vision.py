# Copyright 2025 the Cohere Inc. team. All rights reserved.
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

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Cohere2VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Cohere2VisionForConditionalGeneration`]. It is used to instantiate an
    Cohere2 Vision model according to the specified arguments, defining the model architecture.

    [CohereLabs/command-a-vision-07-2025](https://huggingface.co/CohereLabs/command-a-vision-07-2025)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SiglipVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Cohere2Config`):
            The config object or dictionary of the text backbone.
        downsample_factor (`int`, *optional*, defaults to 2):
            The factor by which to downsample the input image.
        image_token_id (`int`, *optional*, defaults to 255036):
            The token ID to use as placeholder for the image input.
        alignment_intermediate_size (`int`, *optional*, defaults to 36864):
            The size of the intermediate layer for alignment.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
    """

    model_type = "cohere2_vision"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    downsample_factor: int = 2
    image_token_id: int = 255036
    alignment_intermediate_size: int = 36864
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=3072,
                image_size=512,
                num_hidden_layers=27,
                num_attention_heads=12,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "cohere2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["cohere2"](tie_word_embeddings=self.tie_word_embeddings)

        super().__post_init__(**kwargs)


__all__ = ["Cohere2VisionConfig"]
