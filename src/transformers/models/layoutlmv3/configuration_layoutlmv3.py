# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""LayoutLMv3 model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class LayoutLMv3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LayoutLMv3Model`]. It is used to instantiate an
    LayoutLMv3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutLMv3
    [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the LayoutLMv3 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`LayoutLMv3Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`LayoutLMv3Model`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the 2D position embedding might ever be used with. Typically set this to something
            large just in case (e.g., 1024).
        coordinate_size (`int`, *optional*, defaults to `128`):
            Dimension of the coordinate embeddings.
        shape_size (`int`, *optional*, defaults to `128`):
            Dimension of the width and height embeddings.
        has_relative_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a relative attention bias in the self-attention mechanism.
        rel_pos_bins (`int`, *optional*, defaults to 32):
            The number of relative position bins to be used in the self-attention mechanism.
        max_rel_pos (`int`, *optional*, defaults to 128):
            The maximum number of relative positions to be used in the self-attention mechanism.
        max_rel_2d_pos (`int`, *optional*, defaults to 256):
            The maximum number of relative 2D positions in the self-attention mechanism.
        rel_2d_pos_bins (`int`, *optional*, defaults to 64):
            The number of 2D relative position bins in the self-attention mechanism.
        has_spatial_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a spatial attention bias in the self-attention mechanism.
        visual_embed (`bool`, *optional*, defaults to `True`):
            Whether or not to add patch embeddings.
        input_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of the images.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of channels of the images.
        patch_size (`int`, *optional*, defaults to `16`)
            The size (resolution) of the patches.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Example:

    ```python
    >>> from transformers import LayoutLMv3Config, LayoutLMv3Model

    >>> # Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration
    >>> configuration = LayoutLMv3Config()

    >>> # Initializing a model (with random weights) from the microsoft/layoutlmv3-base style configuration
    >>> model = LayoutLMv3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "layoutlmv3"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    max_2d_position_embeddings: int = 1024
    coordinate_size: int = 128
    shape_size: int = 128
    has_relative_attention_bias: bool = True
    rel_pos_bins: int = 32
    max_rel_pos: int = 128
    rel_2d_pos_bins: int = 64
    max_rel_2d_pos: int = 256
    has_spatial_attention_bias: bool = True
    text_embed: bool = True
    visual_embed: bool = True
    input_size: int = 224
    num_channels: int = 3
    patch_size: int = 16
    classifier_dropout: float | int | None = None


__all__ = ["LayoutLMv3Config"]
