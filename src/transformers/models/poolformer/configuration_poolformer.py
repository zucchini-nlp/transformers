# Copyright 2022 Sea AI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""PoolFormer model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class PoolFormerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`PoolFormerModel`]. It is used to instantiate a
    PoolFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PoolFormer
    [sail/poolformer_s12](https://huggingface.co/sail/poolformer_s12) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the input image.
        patch_size (`int`, *optional*, defaults to 16):
            The size of the input patch.
        stride (`int`, *optional*, defaults to 16):
            The stride of the input patch.
        pool_size (`int`, *optional*, defaults to 3):
            The size of the pooling window.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the number of channels in the output of the MLP to the number of channels in the input.
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            The depth of each encoder block.
        hidden_sizes (`list`, *optional*, defaults to `[64, 128, 320, 512]`):
            The hidden sizes of each encoder block.
        patch_sizes (`list`, *optional*, defaults to `[7, 3, 3, 3]`):
            The size of the input patch for each encoder block.
        strides (`list`, *optional*, defaults to `[4, 2, 2, 2]`):
            The stride of the input patch for each encoder block.
        padding (`list`, *optional*, defaults to `[2, 1, 1, 1]`):
            The padding of the input patch for each encoder block.
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            The number of encoder blocks.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout rate for the dropout layers.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the hidden layers.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to use layer scale.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
            The initial value for the layer scale.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The initializer range for the weights.

    Example:

    ```python
    >>> from transformers import PoolFormerConfig, PoolFormerModel

    >>> # Initializing a PoolFormer sail/poolformer_s12 style configuration
    >>> configuration = PoolFormerConfig()

    >>> # Initializing a model (with random weights) from the sail/poolformer_s12 style configuration
    >>> model = PoolFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "poolformer"

    num_channels: int = 3
    patch_size: int = 16
    stride: int = 16
    pool_size: int = 3
    mlp_ratio: float = 4.0
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    hidden_sizes: list[int] | tuple[int, ...] = (64, 128, 320, 512)
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    padding: list[int] | tuple[int, ...] = (2, 1, 1, 1)
    num_encoder_blocks: int = 4
    drop_path_rate: float = 0.0
    hidden_act: str = "gelu"
    use_layer_scale: bool = True
    layer_scale_init_value: float = 1e-5
    initializer_range: float = 0.02


__all__ = ["PoolFormerConfig"]
