# Copyright 2021 The HuggingFace Inc. team.
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
"""OpenAI ImageGPT configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ImageGPTConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a [`ImageGPTModel`]. It is
    used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT
    [openai/imagegpt-small](https://huggingface.co/openai/imagegpt-small) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ImageGPTModel`].
        n_positions (`int`, *optional*, defaults to 32*32):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 512):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"quick_gelu"`):
            Activation function (can be one of the activation functions defined in src/transformers/activations.py).
            Defaults to "quick_gelu".
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
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
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import ImageGPTConfig, ImageGPTModel

    >>> # Initializing a ImageGPT configuration
    >>> configuration = ImageGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ImageGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "imagegpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 512 + 1  # add one for start of sentence (sos) token
    n_positions: int = 32 * 32
    n_embd: int = 512
    n_layer: int = 24
    n_head: int = 8
    n_inner: int | None = None
    activation_function: str = "quick_gelu"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = False
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    add_cross_attention: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


__all__ = ["ImageGPTConfig"]
