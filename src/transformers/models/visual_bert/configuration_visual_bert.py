# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""VisualBERT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class VisualBertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VisualBertModel`]. It is used to instantiate an
    VisualBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the VisualBERT
    [uclanlp/visualbert-vqa-coco-pre](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the VisualBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`VisualBertModel`]. Vocabulary size of the model. Defines the
            different tokens that can be represented by the `inputs_ids` passed to the forward method of
            [`VisualBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        visual_embedding_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the visual embeddings to be passed to the model.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
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
            The vocabulary size of the `token_type_ids` passed when calling [`VisualBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bypass_transformer (`bool`, *optional*, defaults to `False`):
            Whether or not the model should bypass the transformer for the visual embeddings. If set to `True`, the
            model directly concatenates the visual embeddings from [`VisualBertEmbeddings`] with text output from
            transformers, and then pass it to a self-attention layer.
        special_visual_initialize (`bool`, *optional*, defaults to `True`):
            Whether or not the visual token type and position type embedding weights should be initialized the same as
            the textual token type and positive type embeddings. When set to `True`, the weights of the textual token
            type and position type embeddings are copied to the respective visual embedding layers.


    Example:

    ```python
    >>> from transformers import VisualBertConfig, VisualBertModel

    >>> # Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
    >>> configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

    >>> # Initializing a model (with random weights) from the visualbert-vqa-coco-pre style configuration
    >>> model = VisualBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "visual_bert"

    vocab_size: int = 30522
    hidden_size: int = 768
    visual_embedding_dim: int = 512
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
    bypass_transformer: bool = False
    special_visual_initialize: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    tie_word_embeddings: bool = True


__all__ = ["VisualBertConfig"]
