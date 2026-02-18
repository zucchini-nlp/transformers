# Copyright 2022 WeChatAI and The HuggingFace Inc. team. All rights reserved.
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
"""RoCBert model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class RoCBertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RoCBertModel`]. It is used to instantiate a
    RoCBert model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RoCBert
    [weiweishi/roc-bert-base-zh](https://huggingface.co/weiweishi/roc-bert-base-zh) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the RoCBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RoCBertModel`].
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
            The vocabulary size of the `token_type_ids` passed when calling [`RoCBertModel`].
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
        enable_pronunciation (`bool`, *optional*, defaults to `True`):
            Whether or not the model use pronunciation embed when training.
        enable_shape (`bool`, *optional*, defaults to `True`):
            Whether or not the model use shape embed when training.
        pronunciation_embed_dim (`int`, *optional*, defaults to 768):
            Dimension of the pronunciation_embed.
        pronunciation_vocab_size (`int`, *optional*, defaults to 910):
            Pronunciation Vocabulary size of the RoCBert model. Defines the number of different tokens that can be
            represented by the `input_pronunciation_ids` passed when calling [`RoCBertModel`].
        shape_embed_dim (`int`, *optional*, defaults to 512):
            Dimension of the shape_embed.
        shape_vocab_size (`int`, *optional*, defaults to 24858):
            Shape Vocabulary size of the RoCBert model. Defines the number of different tokens that can be represented
            by the `input_shape_ids` passed when calling [`RoCBertModel`].
        concat_input (`bool`, *optional*, defaults to `True`):
            Defines the way of merging the shape_embed, pronunciation_embed and word_embed, if the value is true,
            output_embed = torch.cat((word_embed, shape_embed, pronunciation_embed), -1), else output_embed =
            (word_embed + shape_embed + pronunciation_embed) / 3
        Example:

    ```python
    >>> from transformers import RoCBertModel, RoCBertConfig

    >>> # Initializing a RoCBert weiweishi/roc-bert-base-zh style configuration
    >>> configuration = RoCBertConfig()

    >>> # Initializing a model from the weiweishi/roc-bert-base-zh style configuration
    >>> model = RoCBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "roc_bert"

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
    use_cache: bool = True
    pad_token_id: int | None = 0
    classifier_dropout: float | None = None
    enable_pronunciation: bool = True
    enable_shape: bool = True
    pronunciation_embed_dim: int = 768
    pronunciation_vocab_size: int = 910
    shape_embed_dim: int = 512
    shape_vocab_size: int = 24858
    concat_input: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["RoCBertConfig"]
