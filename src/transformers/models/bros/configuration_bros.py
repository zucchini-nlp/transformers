# Copyright 2023-present NAVER Corp, The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
"""Bros model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BrosConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BrosModel`]. It is used to
    instantiate a Bros model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Bros
    [jinho8345/bros-base-uncased](https://huggingface.co/jinho8345/bros-base-uncased) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Bros model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BrosModel`].
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
            The vocabulary size of the `token_type_ids` passed when calling [`BrosModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The index of the padding token in the token vocabulary.
        dim_bbox (`int`, *optional*, defaults to 8):
            The dimension of the bounding box coordinates. (x0, y1, x1, y0, x1, y1, x0, y1)
        bbox_scale (`float`, *optional*, defaults to 100.0):
            The scale factor of the bounding box coordinates.
        n_relations (`int`, *optional*, defaults to 1):
            The number of relations for SpadeEE(entity extraction), SpadeEL(entity linking) head.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the classifier head.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on
            decoder-only or encoder-only architectures.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model.

    Examples:

    ```python
    >>> from transformers import BrosConfig, BrosModel

    >>> # Initializing a BROS jinho8345/bros-base-uncased style configuration
    >>> configuration = BrosConfig()

    >>> # Initializing a model from the jinho8345/bros-base-uncased style configuration
    >>> model = BrosModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bros"

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
    pad_token_id: int | None = 0
    dim_bbox: int = 8
    bbox_scale: float = 100.0
    n_relations: int = 1
    classifier_dropout_prob: float = 0.1
    is_decoder: bool = False
    add_cross_attention: bool = False

    def __post_init__(self, **kwargs):
        self.dim_bbox_sinusoid_emb_2d = self.hidden_size // 4
        self.dim_bbox_sinusoid_emb_1d = self.dim_bbox_sinusoid_emb_2d // self.dim_bbox
        self.dim_bbox_projection = self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["BrosConfig"]
