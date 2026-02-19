# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""RemBERT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class RemBertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RemBertModel`]. It is used to instantiate an
    RemBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RemBERT
    [google/rembert](https://huggingface.co/google/rembert) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 250300):
            Vocabulary size of the RemBERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RemBertModel`]. Vocabulary size of the model.
            Defines the different tokens that can be represented by the *inputs_ids* passed to the forward method of
            [`RemBertModel`].
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 18):
            Number of attention heads for each attention layer in the Transformer encoder.
        input_embedding_size (`int`, *optional*, defaults to 256):
            Dimensionality of the input embeddings.
        output_embedding_size (`int`, *optional*, defaults to 1664):
            Dimensionality of the output embeddings.
        intermediate_size (`int`, *optional*, defaults to 4608):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0):
            The dropout ratio for the attention probabilities.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the classifier layer when fine-tuning.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`RemBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Example:

    ```python
    >>> from transformers import RemBertModel, RemBertConfig

    >>> # Initializing a RemBERT rembert style configuration
    >>> configuration = RemBertConfig()

    >>> # Initializing a model from the rembert style configuration
    >>> model = RemBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rembert"

    vocab_size: int = 250300
    hidden_size: int = 1152
    num_hidden_layers: int = 32
    num_attention_heads: int = 18
    input_embedding_size: int = 256
    output_embedding_size: int = 1664
    intermediate_size: int = 4608
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    classifier_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 312
    eos_token_id: int | None = 313
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = False


__all__ = ["RemBertConfig"]
