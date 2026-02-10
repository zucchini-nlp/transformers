# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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
"""DeBERTa model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class DebertaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DebertaModel`]. It is
    used to instantiate a DeBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DebertaModel`].
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
            `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
            are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 0):
            The vocabulary size of the `token_type_ids` passed when calling [`DebertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        relative_attention (`bool`, *optional*, defaults to `False`):
            Whether use relative position encoding.
        max_relative_positions (`int`, *optional*, defaults to 1):
            The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
            as `max_position_embeddings`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (`bool`, *optional*, defaults to `True`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (`list[str]`, *optional*):
            The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
            `["p2c", "c2p"]`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        legacy (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the legacy `LegacyDebertaOnlyMLMHead`, which does not work properly
            for mask infilling tasks.

    Example:

    ```python
    >>> from transformers import DebertaConfig, DebertaModel

    >>> # Initializing a DeBERTa microsoft/deberta-base style configuration
    >>> configuration = DebertaConfig()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-base style configuration
    >>> model = DebertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deberta"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-7
    relative_attention: bool = False
    max_relative_positions: int = -1
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    position_biased_input: bool = True
    pos_att_type: str | list[str] | None = None
    pooler_dropout: float = 0.0
    pooler_hidden_act: str = "gelu"
    legacy: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # Backwards compatibility
        if isinstance(self.pos_att_type, str):
            self.pos_att_type = [x.strip() for x in self.pos_att_type.lower().split("|")]

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", self.hidden_size)
        super().__post_init__(**kwargs)


__all__ = ["DebertaConfig"]
