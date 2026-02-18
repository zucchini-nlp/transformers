# Copyright 2025 Westlake Representational Learning Lab (Fajie Yuan Lab) team and the HuggingFace Inc. team. All rights reserved.
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
"""Evolla model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class SaProtConfig(PreTrainedConfig):
    r"""This is the configuration class to store the configuration of a [`EvollaSaProtProteinEncoder`]. It is used to instantiate a
    SaProt model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 446):
            Vocabulary size of the protein sequence model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`EvollaModel`].
        mask_token_id (`int`, *optional*, defaults to 4):
            The id of the *mask* token in the protein sequence model.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the *padding* token in the protein sequence model.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the protein sequence model layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 33):
            Number of hidden layers in the protein sequence model.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the protein sequence model.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the intermediate layers in the protein sequence model.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the hidden layers in the protein sequence model.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in the protein sequence model.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that the protein sequence model might ever be used with. Typically set this to
            something large just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for the layer normalization layer in the protein sequence model.
        position_embedding_type (`str`, *optional*, defaults to `"rotary"`):
            The type of position embedding to use in the protein sequence model. Currently only `"rotary"` is supported.
        emb_layer_norm_before (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization before the position embedding in the protein sequence model.
        token_dropout (`bool`, *optional*, defaults to `True`):
            Whether to apply dropout to the tokens in the protein sequence model."""

    vocab_size: int = 446
    mask_token_id: int = 4
    pad_token_id: int = 1
    hidden_size: int = 1280
    num_hidden_layers: int = 33
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1026
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-05
    position_embedding_type: str = "rotary"
    emb_layer_norm_before: bool = False
    token_dropout: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False


@strict(accept_kwargs=True)
@dataclass(repr=False)
class EvollaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        protein_encoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SaProtConfig`].
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Evolla llama model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`EvollaModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the llama layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the intermediate layers in the llama model.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the llama model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the llama model.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value pairs for each attention layer in the llama model.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the llama model. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"silu"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for the RMS-norm layer in the llama model.
        rope_parameters (`float`, *optional*):
            The scaling factor for the RoPE layer in the llama model.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention layer.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layer.
        aligner_ffn_mult (`int`, *optional*, defaults to 4):
            The FFN multiplier for the aligner layer.
        aligner_enable_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the aligner layer.
        aligner_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in the aligner layer.
        aligner_num_add_layers (`int`, *optional*, defaults to 8):
            The number of additional layers for the aligner layer.
        resampler_depth (`int`, *optional*, defaults to 6):
            The depth of the resampler layer in the llama model.
        resampler_dim_head (`int`, *optional*, defaults to 64):
            The dimension of the heads in the resampler layer in the llama model.
        resampler_heads (`int`, *optional*, defaults to 8):
            The number of heads in the resampler layer in the llama model.
        resampler_num_latents (`int`, *optional*, defaults to 64):
            The number of latents in the resampler layer in the llama model.
        resampler_ff_mult (`int`, *optional*, defaults to 4):
            The FFN multiplier for the resampler layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 128009):
            The id of the *end-of-sequence* token.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the input and output word embeddings.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on
            decoder-only or encoder-only architectures.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model.

    Example:

    ```python
    >>> from transformers import EvollaModel, EvollaConfig

    >>> # Initializing a Evolla evolla-10b style configuration
    >>> configuration = EvollaConfig()

    >>> # Initializing a model from the evolla-10b style configuration
    >>> model = EvollaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "EvollaModel"
    sub_configs = {"protein_encoder_config": SaProtConfig}
    default_theta = 500000.0

    protein_encoder_config: dict | PreTrainedConfig | None = None
    vocab_size: int | None = 128256  # llama vocab size
    hidden_size: int | None = 4096  # llama hidden size
    intermediate_size: int | None = 14336  # llama intermediate size
    num_hidden_layers: int | None = 32  # llama num layers
    num_attention_heads: int | None = 32  # llama num heads
    num_key_value_heads: int | None = 8  # llama num key-value heads
    hidden_act: str | None = "silu"  # llama activation function
    max_position_embeddings: int | None = 8192  # llama rope max length
    rms_norm_eps: float | None = 1e-05
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool | None = False
    attention_dropout: float | int | None = 0.0
    mlp_bias: bool | None = False
    aligner_ffn_mult: int | None = 4
    aligner_enable_bias: bool | None = True
    aligner_attention_probs_dropout_prob: float | None = 0.1
    aligner_num_add_layers: int | None = 8
    resampler_depth: int | None = 6
    resampler_dim_head: int | None = 64
    resampler_heads: int | None = 8
    resampler_num_latents: int | None = 64
    resampler_ff_mult: int | None = 4
    initializer_range: float | None = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128009
    use_cache: bool | None = False
    tie_word_embeddings: bool | None = False
    is_decoder: bool | None = False
    add_cross_attention: bool | None = False

    def __post_init__(self, **kwargs):
        if self.protein_encoder_config is None:
            self.protein_encoder_config = SaProtConfig()
            logger.info("`protein_encoder_config` is `None`. Initializing the `SaProtConfig` with default values.")
        elif isinstance(self.protein_encoder_config, dict):
            self.protein_encoder_config = SaProtConfig(**self.protein_encoder_config)
        super().__post_init__(**kwargs)


__all__ = ["EvollaConfig"]
