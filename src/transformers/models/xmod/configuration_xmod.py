# Copyright 2023 The Meta AI Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""X-MOD configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class XmodConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XmodModel`]. It is used to instantiate an X-MOD
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [facebook/xmod-base](https://huggingface.co/facebook/xmod-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the X-MOD model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`XmodModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`XmodModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization before each block.
        adapter_reduction_factor (`int` or `float`, *optional*, defaults to 2):
            The factor by which the dimensionality of the adapter is reduced relative to `hidden_size`.
        adapter_layer_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply a new layer normalization before the adapter modules (shared across all adapters).
        adapter_reuse_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether to reuse the second layer normalization and apply it before the adapter modules as well.
        ln_before_adapter (`bool`, *optional*, defaults to `True`):
            Whether to apply the layer normalization before the residual connection around the adapter module.
        languages (`Iterable[str]`, *optional*, defaults to `["en_XX"]`):
            An iterable of language codes for which adapter modules should be initialized.
        default_language (`str`, *optional*):
            Language code of a default language. It will be assumed that the input is in this language if no language
            codes are explicitly passed to the forward method.

    Examples:

    ```python
    >>> from transformers import XmodConfig, XmodModel

    >>> # Initializing an X-MOD facebook/xmod-base style configuration
    >>> configuration = XmodConfig()

    >>> # Initializing a model (with random weights) from the facebook/xmod-base style configuration
    >>> model = XmodModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xmod"

    vocab_size: int = 30522
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
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    pre_norm: bool = False
    adapter_reduction_factor: int = 2
    adapter_layer_norm: bool = False
    adapter_reuse_layer_norm: bool = True
    ln_before_adapter: bool = True
    languages: list[str] | tuple[str, ...] = ("en_XX",)
    default_language: str | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["XmodConfig"]
