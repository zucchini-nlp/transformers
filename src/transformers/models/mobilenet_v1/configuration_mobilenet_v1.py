# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""MobileNetV1 model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class MobileNetV1Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileNetV1Model`]. It is used to instantiate a
    MobileNetV1 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileNetV1
    [google/mobilenet_v1_1.0_224](https://huggingface.co/google/mobilenet_v1_1.0_224) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            Shrinks or expands the number of channels in each layer. Default is 1.0, which starts the network with 32
            channels. This is sometimes also called "alpha" or "width multiplier".
        min_depth (`int`, *optional*, defaults to 8):
            All layers will have at least this many channels.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu6"`):
            The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
        tf_padding (`bool`, *optional*, defaults to `True`):
            Whether to use TensorFlow padding rules on the convolution layers.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.999):
            The dropout ratio for attached classifiers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 0.001):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import MobileNetV1Config, MobileNetV1Model

    >>> # Initializing a "mobilenet_v1_1.0_224" style configuration
    >>> configuration = MobileNetV1Config()

    >>> # Initializing a model from the "mobilenet_v1_1.0_224" style configuration
    >>> model = MobileNetV1Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mobilenet_v1"

    num_channels: int = 3
    image_size: int = 224
    depth_multiplier: float = 1.0
    min_depth: int = 8
    hidden_act: str = "relu6"
    tf_padding: bool = True
    classifier_dropout_prob: float = 0.999
    initializer_range: float = 0.02
    layer_norm_eps: float = 0.001

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")


__all__ = ["MobileNetV1Config"]
