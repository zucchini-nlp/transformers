# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
class InternVLVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVLVisionModel`]. It is used to instantiate an InternVLVisionModel
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the InternVL3-1B.
    e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to the queries and keys before the attention operation.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        projection_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the projection layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int` or `list[int]`, *optional*, defaults to `[448, 448]`):
            The size (resolution) of each image.
        patch_size (`int` or `list[int]`, *optional*, defaults to `[14, 14]`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to use BERT-style absolute position embeddings.
        layer_scale_init_value (`float`, *optional*, defaults to 0.1):
            Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
            CLS token, before applying the classification head.

    Example:

    ```python
    >>> from transformers import InternVLVisionConfig, InternVLVisionModel

    >>> # Initializing a InternVLVisionModel OpenGVLab/InternVL3-1B-hf style configuration
    >>> configuration = InternVLVisionConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    attention_bias: bool = False
    use_qk_norm: bool = False
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    initializer_range: float = 0.02
    norm_type: str = "layer_norm"
    layer_norm_eps: float = 1e-06
    image_size: int | list[int] | tuple[int, ...] = (448, 448)
    patch_size: int | list[int] | tuple[int, ...] = (14, 14)
    num_channels: int = 3
    use_mask_token: bool = False
    use_absolute_position_embeddings: bool = True
    layer_scale_init_value: float = 0.1
    use_mean_pooling: bool = True

    def __post_init__(self, **kwargs):
        self.image_size = (
            self.image_size if isinstance(self.image_size, (list, tuple)) else (self.image_size, self.image_size)
        )
        self.patch_size = (
            self.patch_size if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size)
        )
        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class InternVLConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVLForConditionalGeneration`]. It is used to instantiate a
    InternVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of InternVL3-1B.
    e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `InternVisonConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 151667):
            The image token index to encode the image prompt.
        image_seq_length (`int`, *optional*, defaults to 256):
            Number of image tokens to use per image patch.
        downsample_ratio (`float`, *optional*, defaults to 0.5):
            Factor by which to downsample the image.
        projector_hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the projector.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to use as the image features.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings


    ```python
    >>> from transformers import InternVLForConditionalGeneration, InternVLConfig

    >>> # Initializing a InternVL style configuration
    >>> configuration = InternVLConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl"
    sub_configs = {"text_config": AutoConfig, "vision_config": InternVLVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151667
    image_seq_length: int = 256
    downsample_ratio: float = 0.5
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "default"
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = InternVLVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = InternVLVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["InternVLVisionConfig", "InternVLConfig"]
