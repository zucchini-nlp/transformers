# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Idefics3 model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Idefics3VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics3VisionModel`]. It is used to instantiate a
    Idefics3 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SigLIP checkpoint
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) used in the Idefics3 model
    [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer
    >>> from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig

    >>> # Initializing a Idefics3VisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = Idefics3VisionConfig()

    >>> # Initializing a Idefics3VisionTransformer (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = Idefics3VisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics3_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Idefics3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics3Model`]. It is used to instantiate a
    Idefics3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the Idefics3
    [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism. Only
            relevant if `config.is_decoder=True`.
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*, defaults to `IdeficsVisionConfig`):
            Custom vision config or dict for the vision tower
        text_config (`PreTrainedConfig` or `dict`, *optional*, defaults to `LlamaConfig`):
            Custom text config or dict for the text model
        scale_factor (`int`, *optional*, defaults to 2):
            The scale factor for the image encoder.
        pad_token_id (`int`, *optional*, defaults to 128002):
            The id of the padding token.

    Example:
    ```python
    >>> from transformers import Idefics3Model, Idefics3Config
    >>> # Initializing configuration
    >>> configuration = Idefics3Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics3Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics3"
    sub_configs = {"text_config": AutoConfig, "vision_config": Idefics3VisionConfig}

    use_cache: bool = True
    image_token_id: int = 128257
    tie_word_embeddings: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    scale_factor: int = 2
    pad_token_id: int | None = 128_002

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = Idefics3VisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Idefics3VisionConfig(**self.vision_config)

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None, using default Llama text config")
            self.text_config = CONFIG_MAPPING["llama"](
                rms_norm_eps=1e-5,
                pad_token_id=self.pad_token_id,
            )

        super().__post_init__(**kwargs)


__all__ = ["Idefics3Config", "Idefics3VisionConfig"]
