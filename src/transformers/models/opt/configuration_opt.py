# Copyright 2022 The Metaseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""OPT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class OPTConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OPTModel`]. It is used to instantiate a OPT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the OPT
    [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50272):
            Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OPTModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        do_layer_norm_before (`bool`, *optional*, defaults to `True`):
            Whether to perform layer normalization before the attention block.
        word_embed_proj_dim (`int`, *optional*):
            `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
            `hidden_size`.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more
            details.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        enable_bias (`bool`, *optional*, defaults to `True`):
            Whether or not if the linear layers in the attention blocks should use the bias term.
        layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not if the layer norms should have learnable parameters.

    Example:

    ```python
    >>> from transformers import OPTConfig, OPTModel

    >>> # Initializing a OPT facebook/opt-large style configuration
    >>> configuration = OPTConfig()

    >>> # Initializing a model (with random weights) from the facebook/opt-large style configuration
    >>> model = OPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 50272
    hidden_size: int = 768
    num_hidden_layers: int = 12
    ffn_dim: int = 3072
    max_position_embeddings: int = 2048
    do_layer_norm_before: bool = True
    _remove_final_layer_norm: bool = False
    word_embed_proj_dim: int | None = None
    dropout: float = 0.1
    attention_dropout: float = 0.0
    num_attention_heads: int = 12
    activation_function: str = "relu"
    layerdrop: float = 0.0
    init_std: float = 0.02
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 2
    eos_token_id: int | None = 2
    enable_bias: bool = True
    layer_norm_elementwise_affine: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.word_embed_proj_dim = (
            self.word_embed_proj_dim if self.word_embed_proj_dim is not None else self.hidden_size
        )
        super().__post_init__(**kwargs)


__all__ = ["OPTConfig"]
