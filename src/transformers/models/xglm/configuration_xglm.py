# Copyright The HuggingFace Inc. team. All rights reserved.
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
"""XGLM model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class XGLMConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XGLMModel`]. It is used to instantiate an XGLM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the XGLM
    [facebook/xglm-564M](https://huggingface.co/facebook/xglm-564M) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 256008):
            Vocabulary size of the XGLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`XGLMModel`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        d_model (`int`, *optional*, defaults to 1024):
            Dimension of the layers and the pooler layer.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        num_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers Transformer decoder.
        attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, dencoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
            for more details.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import XGLMModel, XGLMConfig

    >>> # Initializing a XGLM facebook/xglm-564M style configuration
    >>> configuration = XGLMConfig()

    >>> # Initializing a model from the facebook/xglm-564M style configuration
    >>> model = XGLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xglm"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    vocab_size: int = 256008
    max_position_embeddings: int = 2048
    d_model: int = 1024
    ffn_dim: int = 4096
    num_layers: int = 24
    attention_heads: int = 16
    activation_function: str = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    layerdrop: float = 0.0
    init_std: float = 0.02
    scale_embedding: bool = True
    use_cache: bool = True
    decoder_start_token_id: int = 2
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["XGLMConfig"]
