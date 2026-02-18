# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""SAM2 model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam2HieraDetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2HieraDetModel`]. It is used to instantiate
    a HieraDet model as defined in the original sam2 repo according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 96):
            The hidden dimension of the image encoder.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the image.
        image_size (`list[int]`, *optional*, defaults to `[1024, 1024]`):
            The size of the image.
        patch_kernel_size (`list[int]`, *optional*, defaults to `[7, 7]`):
            The kernel size of the patch.
        patch_stride (`list[int]`, *optional*, defaults to `[4, 4]`):
            The stride of the patch.
        patch_padding (`list[int]`, *optional*, defaults to `[3, 3]`):
            The padding of the patch.
        query_stride (`list[int]`, *optional*, defaults to `[2, 2]`):
            The downsample stride between stages.
        window_positional_embedding_background_size (`list[int]`, *optional*, defaults to `[7, 7]`):
            The window size per stage when not using global attention.
        num_query_pool_stages (`int`, *optional*, defaults to 3):
            The number of query pool stages.
        blocks_per_stage (`list[int]`, *optional*, defaults to `[1, 2, 7, 2]`):
            The number of blocks per stage.
        embed_dim_per_stage (`list[int]`, *optional*, defaults to `[96, 192, 384, 768]`):
            The embedding dimension per stage.
        num_attention_heads_per_stage (`list[int]`, *optional*, defaults to `[1, 2, 4, 8]`):
            The number of attention heads per stage.
        window_size_per_stage (`list[int]`, *optional*, defaults to `[8, 4, 14, 7]`):
            The window size per stage.
        global_attention_blocks (`list[int]`, *optional*, defaults to `[5, 7, 9]`):
            The blocks where global attention is used.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the MLP hidden dimension to the embedding dimension.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "backbone_config"
    model_type = "sam2_hiera_det_model"

    hidden_size: int = 96
    num_attention_heads: int = 1
    num_channels: int = 3
    image_size: int | list[int] | None = None
    patch_kernel_size: int | list[int] | None = None
    patch_stride: int | list[int] | None = None
    patch_padding: int | list[int] | None = None
    query_stride: int | list[int] | None = None
    window_positional_embedding_background_size: list[int] | None = None
    num_query_pool_stages: int = 3
    blocks_per_stage: list[int] | None = None
    embed_dim_per_stage: list[int] | None = None
    num_attention_heads_per_stage: list[int] | None = None
    window_size_per_stage: list[int] | None = None
    global_attention_blocks: list[int] | None = None
    mlp_ratio: float = 4.0
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.image_size = self.image_size if self.image_size is not None else [1024, 1024]
        self.patch_kernel_size = self.patch_kernel_size if self.patch_kernel_size is not None else [7, 7]
        self.patch_stride = self.patch_stride if self.patch_stride is not None else [4, 4]
        self.patch_padding = self.patch_padding if self.patch_padding is not None else [3, 3]
        self.query_stride = self.query_stride if self.query_stride is not None else [2, 2]
        self.window_positional_embedding_background_size = (
            self.window_positional_embedding_background_size
            if self.window_positional_embedding_background_size is not None
            else [7, 7]
        )
        self.blocks_per_stage = self.blocks_per_stage if self.blocks_per_stage is not None else [1, 2, 7, 2]
        self.embed_dim_per_stage = (
            self.embed_dim_per_stage if self.embed_dim_per_stage is not None else [96, 192, 384, 768]
        )
        self.num_attention_heads_per_stage = (
            self.num_attention_heads_per_stage if self.num_attention_heads_per_stage is not None else [1, 2, 4, 8]
        )
        self.window_size_per_stage = (
            self.window_size_per_stage if self.window_size_per_stage is not None else [8, 4, 14, 7]
        )
        self.global_attention_blocks = (
            self.global_attention_blocks if self.global_attention_blocks is not None else [5, 7, 9]
        )
        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam2VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2VisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `Sam2HieraDetConfig()`):
            Configuration for the vision backbone. This is used to instantiate the backbone using
            `AutoModel.from_config`.
        backbone_channel_list (`List[int]`, *optional*, defaults to `[768, 384, 192, 96]`):
            The list of channel dimensions for the backbone.
        backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[256, 256], [128, 128], [64, 64]]`):
            The spatial sizes of the feature maps from the backbone.
        fpn_hidden_size (`int`, *optional*, defaults to 256):
            The hidden dimension of the FPN.
        fpn_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size for the convolutions in the neck.
        fpn_stride (`int`, *optional*, defaults to 1):
            The stride for the convolutions in the neck.
        fpn_padding (`int`, *optional*, defaults to 0):
            The padding for the convolutions in the neck.
        fpn_top_down_levels (`List[int]`, *optional*, defaults to `[2, 3]`):
            The levels for the top-down FPN connections.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of feature levels from the FPN to use.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the neck.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon for the layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    """

    base_config_key = "vision_config"
    model_type = "sam2_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    backbone_config: dict | PreTrainedConfig | None = None
    backbone_channel_list: list[int] | None = None
    backbone_feature_sizes: list | None = None
    fpn_hidden_size: int = 256
    fpn_kernel_size: int = 1
    fpn_stride: int = 1
    fpn_padding: int = 0
    fpn_top_down_levels: list[int] | None = None
    num_feature_levels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.backbone_channel_list = (
            [768, 384, 192, 96] if self.backbone_channel_list is None else self.backbone_channel_list
        )
        self.backbone_feature_sizes = (
            [[256, 256], [128, 128], [64, 64]] if self.backbone_feature_sizes is None else self.backbone_feature_sizes
        )
        self.fpn_top_down_levels = [2, 3] if self.fpn_top_down_levels is None else self.fpn_top_down_levels

        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "sam2_hiera_det_model")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = Sam2HieraDetConfig()

        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam2PromptEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2PromptEncoder`]. The [`Sam2PromptEncoder`]
    module is used to encode the input 2D points and bounding boxes.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1024):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        scale (`float`, *optional*, defaults to 1):
            The scale factor for the prompt encoder.
    """

    base_config_key = "prompt_encoder_config"

    hidden_size: int = 256
    image_size: int = 1024
    patch_size: int = 16
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    scale: int = 1


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam2MaskDecoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sam2MaskDecoder`]. It is used to instantiate a SAM2
    memory encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the SAM2 mask decoder.
        mlp_dim (`int`, *optional*, defaults to 2048):
            The dimension of the MLP in the two-way transformer.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            The number of hidden layers in the two-way transformer.
        num_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads in the two-way transformer.
        attention_downsample_rate (`int`, *optional*, defaults to 2):
            The downsample rate for the attention layers.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            The number of multimask outputs.
        iou_head_depth (`int`, *optional*, defaults to 3):
            The depth of the IoU head.
        iou_head_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the IoU head.
        dynamic_multimask_via_stability (`bool`, *optional*, defaults to `True`):
            Whether to use dynamic multimask via stability.
        dynamic_multimask_stability_delta (`float`, *optional*, defaults to 0.05):
            The stability delta for the dynamic multimask.
        dynamic_multimask_stability_thresh (`float`, *optional*, defaults to 0.98):
            The stability threshold for the dynamic multimask.

    """

    base_config_key = "mask_decoder_config"

    hidden_size: int = 256
    hidden_act: str = "gelu"
    mlp_dim: int = 2048
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    dynamic_multimask_via_stability: bool = True
    dynamic_multimask_stability_delta: float = 0.05
    dynamic_multimask_stability_thresh: float = 0.98


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Sam2Config(PreTrainedConfig):
    r"""
    [`Sam2Config`] is the configuration class to store the configuration of a [`Sam2Model`]. It is used to instantiate a
    SAM2 model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
    configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2.1 Hiera-tiny
    [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    <Tip>

    SAM2 checkpoints with `model_type="sam2_video"` are compatible with `Sam2Model` since the video variant weights
    are a superset of the image-only model weights. You may see a warning about model type mismatch when loading
    such checkpoints, which can be safely ignored in this case.

    </Tip>

    Args:
        vision_config (Union[`dict`, `Sam2VisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2VisionConfig`].
        prompt_encoder_config (Union[`dict`, `Sam2PromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2PromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `Sam2MaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`Sam2MaskDecoderConfig`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for parameter initialization.

    Example:

    ```python
    >>> from transformers import (
    ...     Sam2VisionConfig,
    ...     Sam2PromptEncoderConfig,
    ...     Sam2MaskDecoderConfig,
    ...     Sam2Model,
    ... )

    >>> # Initializing a Sam2Config with `"facebook/sam2.1_hiera_tiny"` style configuration
    >>> configuration = Sam2Config()

    >>> # Initializing a Sam2Model (with random weights) from the `"facebook/sam2.1_hiera_tiny"` style configuration
    >>> model = Sam2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Sam2Config from a Sam2VisionConfig, Sam2PromptEncoderConfig, and Sam2MaskDecoderConfig

    >>> # Initializing SAM2 vision encoder, memory attention, and memory encoder configurations
    >>> vision_config = Sam2VisionConfig()
    >>> prompt_encoder_config = Sam2PromptEncoderConfig()
    >>> mask_decoder_config = Sam2MaskDecoderConfig()

    >>> config = Sam2Config(vision_config, prompt_encoder_config, mask_decoder_config)
    ```"""

    model_type = "sam2"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam2PromptEncoderConfig,
        "mask_decoder_config": Sam2MaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam2_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam2_vision_model"]()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = Sam2PromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = Sam2PromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam2MaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = Sam2MaskDecoderConfig()

        super().__post_init__(**kwargs)


__all__ = [
    "Sam2Config",
    "Sam2HieraDetConfig",
    "Sam2VisionConfig",
    "Sam2PromptEncoderConfig",
    "Sam2MaskDecoderConfig",
]
