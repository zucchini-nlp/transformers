# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""FocalNet model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class FocalNetConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FocalNetModel`]. It is used to instantiate a
    FocalNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the FocalNet
    [microsoft/focalnet-tiny](https://huggingface.co/microsoft/focalnet-tiny) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch in the embeddings layer.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        use_conv_embed (`bool`, *optional*, defaults to `False`):
            Whether to use convolutional embedding. The authors noted that using convolutional embedding usually
            improve the performance, but it's not used by default.
        hidden_sizes (`list[int]`, *optional*, defaults to `[192, 384, 768, 768]`):
            Dimensionality (hidden size) at each stage.
        depths (`list(int)`, *optional*, defaults to `[2, 2, 6, 2]`):
            Depth (number of layers) of each stage in the encoder.
        focal_levels (`list(int)`, *optional*, defaults to `[2, 2, 2, 2]`):
            Number of focal levels in each layer of the respective stages in the encoder.
        focal_windows (`list(int)`, *optional*, defaults to `[3, 3, 3, 3]`):
            Focal window size in each layer of the respective stages in the encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        use_layerscale (`bool`, *optional*, defaults to `False`):
            Whether to use layer scale in the encoder.
        layerscale_value (`float`, *optional*, defaults to 0.0001):
            The initial value of the layer scale.
        use_post_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to use post layer normalization in the encoder.
        use_post_layernorm_in_modulation (`bool`, *optional*, defaults to `False`):
            Whether to use post layer normalization in the modulation layer.
        normalize_modulator (`bool`, *optional*, defaults to `False`):
            Whether to normalize the modulator.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        encoder_stride (`int`, *optional*, defaults to 32):
            Factor to increase the spatial resolution by in the decoder head for masked image modeling.
        out_features (`list[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.

    Example:

    ```python
    >>> from transformers import FocalNetConfig, FocalNetModel

    >>> # Initializing a FocalNet microsoft/focalnet-tiny style configuration
    >>> configuration = FocalNetConfig()

    >>> # Initializing a model (with random weights) from the microsoft/focalnet-tiny style configuration
    >>> model = FocalNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "focalnet"

    image_size: int = 224
    patch_size: int = 4
    num_channels: int = 3
    embed_dim: int = 96
    use_conv_embed: bool = False
    hidden_sizes: list[int] | tuple[int, ...] = (192, 384, 768, 768)
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    focal_levels: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    focal_windows: list[int] | tuple[int, ...] = (3, 3, 3, 3)
    hidden_act: str = "gelu"
    mlp_ratio: float = 4.0
    hidden_dropout_prob: float = 0.0
    drop_path_rate: float = 0.1
    use_layerscale: bool = False
    layerscale_value: float = 1e-4
    use_post_layernorm: bool = False
    use_post_layernorm_in_modulation: bool = False
    normalize_modulator: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    encoder_stride: int = 32
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["FocalNetConfig"]
