# Copyright 2023 The BigCode team and HuggingFace Inc. team.
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
"""GPTBigCode configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class GPTBigCodeConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPTBigCodeModel`]. It is used to instantiate a
    GPTBigCode model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTBigCode
    [gpt_bigcode](https://huggingface.co/gpt_bigcode) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTBigCodeModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new",
            "gelu_pytorch_tanh"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to call the fused softmax in float32.
        scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to scale the attention softmax in float32.
        attention_type (`bool`, *optional*, defaults to `True`):
            Whether to use Multi-Query Attion (`True`) or Multi-Head Attention (`False`).
    Example:

    ```python
    >>> from transformers import GPTBigCodeConfig, GPTBigCodeModel

    >>> # Initializing a GPTBigCode configuration
    >>> configuration = GPTBigCodeConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPTBigCodeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt_bigcode"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    activation_function: str = "gelu_pytorch_tanh"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | None = 50256
    pad_token_id: int | None = None
    attention_softmax_in_fp32: bool = True
    scale_attention_softmax_in_fp32: bool = True
    multi_query: bool = True
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = 1 if self.multi_query else self.n_head
        super().__post_init__(**kwargs)


__all__ = ["GPTBigCodeConfig"]
