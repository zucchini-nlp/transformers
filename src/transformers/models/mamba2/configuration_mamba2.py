# Copyright 2024 The HuggingFace Inc. team.
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
"""MAMBA2 configuration"""

import math
from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Mamba2Config(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Mamba2Model`]. It is used to instantiate a MAMBA2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA2
    [state-spaces/mamba2-2.8b](https://huggingface.co/state-spaces/mamba2-2.8b) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        num_heads (`int`, *optional*, defaults to 128):
            Number of heads for the evolution matrices of mamba 2.
        head_dim (`int`, *optional*, defaults to 64):
            Dimension of each head.
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the MAMBA2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Mamba2Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 128): shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end of sentence token in the vocabulary.
        expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        n_groups (`int`, *optional*, defaults to 8):
            Number of groups for the evolution matrices of mamba 2.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`tuple`, *optional*, defaults to `(0.0, inf)`):
            Accepted range of time step values.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        rms_norm (`bool`, *optional*, defaults to `True`):
            Whether to use RMS norm or not.
        chunk_size (`int`, *optional*, defaults to 256):
            Size of the chunks that will comprise the sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings or not.


    Example:

    ```python
    >>> from transformers import Mamba2Config, Mamba2Model

    >>> # Initializing a Mamba2 configuration
    >>> configuration = Mamba2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Mamba2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mamba2"

    num_heads: int = 128
    head_dim: int = 64
    vocab_size: int = 32768
    hidden_size: int = 4096
    state_size: int = 128
    num_hidden_layers: int = 64
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str | int = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: list[float] | tuple[float, ...] = (0.0, float("inf"))
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    rms_norm: bool = True
    chunk_size: int = 256
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.time_step_rank = (
            math.ceil(self.hidden_size / 16) if self.time_step_rank == "auto" else self.time_step_rank
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if (self.hidden_size * self.expand) != (self.num_heads * self.head_dim):
            raise ValueError(
                "Inconsistent configuration: hidden_size * expand "
                f"({self.hidden_size * self.expand}) must equal num_heads * head_dim "
                f"({self.num_heads * self.head_dim})."
            )


__all__ = ["Mamba2Config"]
