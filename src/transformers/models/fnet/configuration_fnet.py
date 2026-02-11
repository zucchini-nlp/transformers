# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""FNet model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class FNetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FNetModel`]. It is used to instantiate an FNet
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FNet
    [google/fnet-base](https://huggingface.co/google/fnet-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the FNet model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FNetModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 4):
            The vocabulary size of the `token_type_ids` passed when calling [`FNetModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_tpu_fourier_optimizations (`bool`, *optional*, defaults to `False`):
            Determines whether to use TPU optimized FFTs. If `True`, the model will favor axis-wise FFTs transforms.
            Set to `False` for GPU/CPU hardware, in which case n-dimensional FFTs are used.
        tpu_short_seq_length (`int`, *optional*, defaults to 512):
            The sequence length that is expected by the model when using TPUs. This will be used to initialize the DFT
            matrix only when *use_tpu_fourier_optimizations* is set to `True` and the input sequence is shorter than or
            equal to 4096 tokens.

    Example:

    ```python
    >>> from transformers import FNetConfig, FNetModel

    >>> # Initializing a FNet fnet-base style configuration
    >>> configuration = FNetConfig()

    >>> # Initializing a model (with random weights) from the fnet-base style configuration
    >>> model = FNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fnet"

    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 4
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_tpu_fourier_optimizations: bool = False
    tpu_short_seq_length: int = 512
    pad_token_id: int | None = 3
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2
    tie_word_embeddings: bool = True


__all__ = ["FNetConfig"]
