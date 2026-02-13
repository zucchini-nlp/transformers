# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""LeViT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class LevitConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LevitModel`]. It is used to instantiate a LeViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LeViT
    [facebook/levit-128S](https://huggingface.co/facebook/levit-128S) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size of the input image.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the initial convolution layers of patch embedding.
        stride (`int`, *optional*, defaults to 2):
            The stride size for the initial convolution layers of patch embedding.
        padding (`int`, *optional*, defaults to 1):
            The padding size for the initial convolution layers of patch embedding.
        patch_size (`int`, *optional*, defaults to 16):
            The patch size for embeddings.
        hidden_sizes (`list[int]`, *optional*, defaults to `[128, 256, 384]`):
            Dimension of each of the encoder blocks.
        num_attention_heads (`list[int]`, *optional*, defaults to `[4, 8, 12]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        depths (`list[int]`, *optional*, defaults to `[4, 4, 4]`):
            The number of layers in each encoder block.
        key_dim (`list[int]`, *optional*, defaults to `[16, 16, 16]`):
            The size of key in each of the encoder blocks.
        drop_path_rate (`int`, *optional*, defaults to 0):
            The dropout probability for stochastic depths, used in the blocks of the Transformer encoder.
        mlp_ratios (`list[int]`, *optional*, defaults to `[2, 2, 2]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_ratios (`list[int]`, *optional*, defaults to `[2, 2, 2]`):
            Ratio of the size of the output dimension compared to input dimension of attention layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import LevitConfig, LevitModel

    >>> # Initializing a LeViT levit-128S style configuration
    >>> configuration = LevitConfig()

    >>> # Initializing a model (with random weights) from the levit-128S style configuration
    >>> model = LevitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "levit"

    image_size: int = 224
    num_channels: int = 3
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    patch_size: int = 16
    hidden_sizes: list[int] | tuple[int, ...] = (128, 256, 384)
    num_attention_heads: list[int] | tuple[int, ...] = (4, 8, 12)
    depths: list[int] | tuple[int, ...] = (4, 4, 4)
    key_dim: list[int] | tuple[int, ...] = (16, 16, 16)
    drop_path_rate: int = 0
    mlp_ratio: list[int] | tuple[int, ...] = (2, 2, 2)
    attention_ratio: list[int] | tuple[int, ...] = (2, 2, 2)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.down_ops = [
            ["Subsample", self.key_dim[0], self.hidden_sizes[0] // self.key_dim[0], 4, 2, 2],
            ["Subsample", self.key_dim[0], self.hidden_sizes[1] // self.key_dim[0], 4, 2, 2],
        ]
        super().__post_init__(**kwargs)


__all__ = ["LevitConfig"]
