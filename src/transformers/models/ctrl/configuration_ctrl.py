# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Salesforce CTRL configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class CTRLConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CTRLModel`]. It is used to
    instantiate a CTRL model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Salesforce/ctrl](https://huggingface.co/Salesforce/ctrl) architecture from SalesForce.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 246534):
            Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CTRLModel`].
        n_positions (`int`, *optional*, defaults to 256):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 1280):
            Dimensionality of the embeddings and hidden states.
        dff (`int`, *optional*, defaults to 8192):
            Dimensionality of the inner dimension of the feed forward networks (FFN).
        n_layer (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-06):
            The epsilon to use in the layer normalization layers
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings

    Examples:

    ```python
    >>> from transformers import CTRLConfig, CTRLModel

    >>> # Initializing a CTRL configuration
    >>> configuration = CTRLConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CTRLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ctrl"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 246534
    n_positions: int = 256
    n_embd: int = 1280
    dff: int = 8192
    n_layer: int = 48
    n_head: int = 16
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True


__all__ = ["CTRLConfig"]
