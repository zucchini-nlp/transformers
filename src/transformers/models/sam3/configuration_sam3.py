# Copyright 2025 Meta AI and The HuggingFace Team. All rights reserved.
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
"""SAM3 model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from transformers import CLIPTextConfig

from ...configuration_utils import PreTrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam3ViTConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Vision Encoder (ViT backbone).

    Instantiating a configuration defaults will yield a similar configuration to that of SAM 3
    [facebook/sam3](https://huggingface.co/facebook/sam3) architecture.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers.
        intermediate_size (`int`, *optional*, defaults to 4736):
            Dimensionality of the feedforward (MLP) layers.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        image_size (`int`, *optional*, defaults to 1008):
            Expected input image size.
        patch_size (`int`, *optional*, defaults to 14):
            Size of image patches.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for RoPE.
        window_size (`int`, *optional*, defaults to 24):
            Window size for windowed attention.
        global_attn_indexes (`list[int]`, *optional*, defaults to `[7, 15, 23, 31]`):
            Indexes of layers with global attention.
        layer_scale_init_value (`float`, *optional*):
            Initial value for layer scale. None means no layer scale.
        pretrain_image_size (`int`, *optional*, defaults to 336):
            Pretrained model image size for position embedding initialization.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    base_config_key = "backbone_config"
    model_type = "sam3_vit_model"

    hidden_size: int = 1024
    intermediate_size: int = 4736
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 1008
    patch_size: int = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    window_size: int = 24
    global_attn_indexes: list[int] | None = None
    layer_scale_init_value: float | None = None
    pretrain_image_size: int = 336
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.global_attn_indexes is None:
            self.global_attn_indexes = [7, 15, 23, 31]


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam3VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam3VisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of SAM 3
    [facebook/sam3](https://huggingface.co/facebook/sam3) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `Sam3ViTConfig()`):
            Configuration for the vision backbone. This is used to instantiate the backbone using
            `AutoModel.from_config`.
        fpn_hidden_size (`int`, *optional*, defaults to 256):
            The hidden dimension of the FPN.
        backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[288, 288], [144, 144], [72, 72]]`):
            The spatial sizes (height, width) of the feature maps from the backbone at different scales.
        scale_factors (`list[float]`, *optional*, defaults to `[4.0, 2.0, 1.0, 0.5]`):
            Scale factors for FPN multi-scale features. List of scaling factors for each FPN level.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "vision_config"
    model_type = "sam3_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    backbone_config: dict | PreTrainedConfig | None = None
    fpn_hidden_size: int = 256
    backbone_feature_sizes: list | None = None
    scale_factors: list[float] | None = None
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.scale_factors = [4.0, 2.0, 1.0, 0.5] if self.scale_factors is None else self.scale_factors
        if self.backbone_feature_sizes is None:
            self.backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]

        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "sam3_vit_model")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = CONFIG_MAPPING["sam3_vit_model"]()

        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the vision encoder."""
        return self.backbone_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to backbone."""
        self.backbone_config.image_size = value


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam3GeometryEncoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Geometry Encoder.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        num_layers (`int`, *optional*, defaults to 3):
            Number of transformer encoder layers for processing geometry prompts.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the geometry encoder.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the feedforward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        roi_size (`int`, *optional*, defaults to 7):
            ROI size for box pooling operations.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_geometry_encoder"

    hidden_size: int = 256
    num_layers: int = 3
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    roi_size: int = 7
    initializer_range: float = 0.02


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam3DETREncoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 DETR Encoder (vision-text fusion encoder).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        num_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the feedforward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_detr_encoder"

    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam3DETRDecoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 DETR Decoder (object query decoder).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the decoder layers.
        num_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        num_queries (`int`, *optional*, defaults to 200):
            Number of object queries.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the feedforward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function in FFN.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for hidden states.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_detr_decoder"

    hidden_size: int = 256
    num_layers: int = 6
    num_queries: int = 200
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam3MaskDecoderConfig(PreTrainedConfig):
    r"""
    Configuration class for SAM3 Mask Decoder (pixel-level mask prediction).

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the mask decoder.
        num_upsampling_stages (`int`, *optional*, defaults to 3):
            Number of upsampling stages in the pixel decoder (FPN).
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for layer normalization.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for prompt cross-attention.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for prompt cross-attention.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
    """

    model_type = "sam3_mask_decoder"

    hidden_size: int = 256
    num_upsampling_stages: int = 3
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    num_attention_heads: int = 8
    initializer_range: float = 0.02


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam3Config(PreTrainedConfig):
    r"""
    Configuration class to store the configuration of a [`Sam3Model`].

    Instantiating a configuration defaults will yield a similar configuration to that of SAM 3
    [facebook/sam3](https://huggingface.co/facebook/sam3) architecture.

    This is the main configuration class that combines all sub-configurations for the SAM3 model.

    <Tip>

    SAM3 checkpoints with `model_type="sam3_video"` are compatible with `Sam3Model` since the video variant weights
    are a superset of the image-only model weights. You may see a warning about model type mismatch when loading
    such checkpoints, which can be safely ignored in this case.

    </Tip>

    Args:
        vision_config (`dict` or `Sam3VisionConfig`, *optional*):
            Configuration for the vision encoder.
        text_config (`dict` or `Sam3TextConfig`, *optional*):
            Configuration for the text encoder.
        geometry_encoder_config (`dict` or `Sam3GeometryEncoderConfig`, *optional*):
            Configuration for the geometry encoder.
        detr_encoder_config (`dict` or `Sam3DETREncoderConfig`, *optional*):
            Configuration for the DETR encoder.
        detr_decoder_config (`dict` or `Sam3DETRDecoderConfig`, *optional*):
            Configuration for the DETR decoder.
        mask_decoder_config (`dict` or `Sam3MaskDecoderConfig`, *optional*):
            Configuration for the mask decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.

    Example:
    ```python
    >>> from transformers import Sam3Config, Sam3Model

    >>> # Initializing a SAM3 configuration
    >>> configuration = Sam3Config()

    >>> # Initializing a model from the configuration
    >>> model = Sam3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "sam3"
    is_composition = True
    sub_configs = {
        "vision_config": Sam3VisionConfig,
        "text_config": CLIPTextConfig,
        "geometry_encoder_config": Sam3GeometryEncoderConfig,
        "detr_encoder_config": Sam3DETREncoderConfig,
        "detr_decoder_config": Sam3DETRDecoderConfig,
        "mask_decoder_config": Sam3MaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    geometry_encoder_config: dict | PreTrainedConfig | None = None
    detr_encoder_config: dict | PreTrainedConfig | None = None
    detr_decoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = Sam3VisionConfig()
        if isinstance(self.vision_config, dict):
            self.vision_config = Sam3VisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = CLIPTextConfig(
                **{
                    "vocab_size": 49408,
                    "hidden_size": 1024,
                    "intermediate_size": 4096,  # hidden_size * mlp_ratio (1024 * 4)
                    "projection_dim": 512,  # CLIP's internal projection dimension
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "max_position_embeddings": 32,
                    "hidden_act": "gelu",
                }
            )
        if isinstance(self.text_config, dict):
            self.text_config = CLIPTextConfig(**self.text_config)

        if self.geometry_encoder_config is None:
            self.geometry_encoder_config = Sam3GeometryEncoderConfig()
        if isinstance(self.geometry_encoder_config, dict):
            self.geometry_encoder_config = Sam3GeometryEncoderConfig(**self.geometry_encoder_config)

        if self.detr_encoder_config is None:
            self.detr_encoder_config = Sam3DETREncoderConfig()
        if isinstance(self.detr_encoder_config, dict):
            self.detr_encoder_config = Sam3DETREncoderConfig(**self.detr_encoder_config)

        if self.detr_decoder_config is None:
            self.detr_decoder_config = Sam3DETRDecoderConfig()
        if isinstance(self.detr_decoder_config, dict):
            self.detr_decoder_config = Sam3DETRDecoderConfig(**self.detr_decoder_config)

        if self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3MaskDecoderConfig()
        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3MaskDecoderConfig(**self.mask_decoder_config)

        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        """Image size for the SAM3 model."""
        return self.vision_config.image_size

    @image_size.setter
    def image_size(self, value):
        """Set the image size and propagate to vision config."""
        self.vision_config.image_size = value


__all__ = [
    "Sam3Config",
    "Sam3ViTConfig",
    "Sam3VisionConfig",
    "Sam3GeometryEncoderConfig",
    "Sam3DETREncoderConfig",
    "Sam3DETRDecoderConfig",
    "Sam3MaskDecoderConfig",
]
