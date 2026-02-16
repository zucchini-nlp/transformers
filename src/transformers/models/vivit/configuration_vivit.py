# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""ViViT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class VivitConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VivitModel`]. It is used to instantiate a ViViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViViT
    [google/vivit-b-16x2-kinetics400](https://huggingface.co/google/vivit-b-16x2-kinetics400) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        num_frames (`int`, *optional*, defaults to 32):
            The number of frames in each video.
        tubelet_size (`list[int]`, *optional*, defaults to `[2, 16, 16]`):
            The size (resolution) of each tubelet.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_fast"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"gelu_fast"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.

    Example:

    ```python
    >>> from transformers import VivitConfig, VivitModel

    >>> # Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
    >>> configuration = VivitConfig()

    >>> # Initializing a model (with random weights) from the google/vivit-b-16x2-kinetics400 style configuration
    >>> model = VivitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vivit"

    image_size: int = 224
    num_frames: int = 32
    tubelet_size: list[int] | tuple[int, ...] = (2, 16, 16)
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_fast"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-06
    qkv_bias: bool = True


__all__ = ["VivitConfig"]
