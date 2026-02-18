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

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, PretrainedConfig
from ...modeling_rope_utils import RopeParameters
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper import TimmWrapperConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class PeVideoEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PeVideoEncoder`]. It is used to instantiate a
    PeVideoEncoder model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of pe-av-large.
    e.g. [facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vision_config (`Union[PreTrainedConfig, dict]`, *optional*):
            Configuration for the vision backbone used to extract frame embeddings. If a dictionary is provided, it is
            used to instantiate a [`~transformers.TimmWrapperConfig`] with the PE default arguments.
        hidden_size (`int`, *optional*, defaults to 1792):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4800):
            Dimension of the feedforward layers in the Transformer blocks.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of Transformer encoder blocks.
        num_attention_heads (`int`, *optional*, defaults to 14):
            Number of attention heads used in each attention layer.
        num_key_value_heads (`int`, *optional*):
            Number of key and value heads for grouped-query attention. If unset, this defaults to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head for query, key, and value projections.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the Transformer blocks.
        max_position_embeddings (`int`, *optional*, defaults to 10000):
            Maximum sequence length supported by the rotary position embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer for weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon used by the RMS normalization layers.
        rope_parameters (`Union[RopeParameters, dict]`, *optional*, defaults to `{'rope_theta': 20000}`):
            Parameters for the rotary position embeddings, such as the base `rope_theta`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias terms in the query, key, value, and output projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio applied to attention probabilities.

    ```python
    >>> from transformers import PeAudioEncoder, PeAudioEncoderConfig

    >>> # Initializing a PeAudioEncoder style configuration
    >>> configuration = PeAudioEncoderConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_video_encoder"
    sub_configs = {"vision_config": TimmWrapperConfig}
    base_config_key = "audio_video_config"

    _default_vision_config_kwargs = {
        "architecture": "vit_pe_core_large_patch14_336",
        "do_pooling": True,
        "num_classes": 1024,
        "global_pool": "map",
        "initializer_range": 0.02,
    }

    vision_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 1792
    intermediate_size: int = 4800
    num_hidden_layers: int = 6
    num_attention_heads: int = 14
    num_key_value_heads: int | None = None
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 10000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 20000}

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "timm_wrapper")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]].from_dict(
                {**self._default_vision_config_kwargs, **self.vision_config}
            )
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["timm_wrapper"].from_dict(self._default_vision_config_kwargs)

        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class PeVideoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PeVideoModel`]. It is used to instantiate a
    PeVideoModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of pe-av-large.
    e.g. [facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`dict` or `PreTrainedConfig`, *optional*):
            Configuration for the text model component.
        video_config (`dict` or `PreTrainedConfig`, *optional*):
            Configuration for the video encoder component.

    ```python
    >>> from transformers import PeVideoModel, PeVideoConfig

    >>> # Initializing a PeVideoModel style configuration
    >>> configuration = PeVideoConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeVideoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_video"
    sub_configs = {"text_config": AutoConfig, "video_config": PeVideoEncoderConfig}
    base_config_key = "audio_video_config"

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
    }

    text_config: dict | PreTrainedConfig | None = None
    video_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "modernbert")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(self.video_config, dict):
            self.video_config = PeVideoEncoderConfig(**self.video_config)
        elif self.video_config is None:
            self.video_config = PeVideoEncoderConfig()

        super().__post_init__(**kwargs)


__all__ = ["PeVideoEncoderConfig", "PeVideoConfig"]
