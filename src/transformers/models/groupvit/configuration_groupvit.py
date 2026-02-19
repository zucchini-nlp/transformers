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
"""GroupViT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class GroupViTTextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GroupViTTextModel`]. It is used to instantiate an
    GroupViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GroupViT
    [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the GroupViT text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`GroupViTModel`].
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import GroupViTTextConfig, GroupViTTextModel

    >>> # Initializing a GroupViTTextModel with nvidia/groupvit-gcc-yfcc style configuration
    >>> configuration = GroupViTTextConfig()

    >>> model = GroupViTTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "groupvit_text_model"
    base_config_key = "text_config"

    vocab_size: int = 49408
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 49406
    eos_token_id: int | None = 49407


@strict(accept_kwargs=True)
@dataclass(repr=False)
class GroupViTVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GroupViTVisionModel`]. It is used to instantiate
    an GroupViT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GroupViT
    [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        depths (`list[int]`, *optional*, defaults to [6, 3, 3]):
            The number of layers in each encoder block.
        num_group_tokens (`list[int]`, *optional*, defaults to [64, 8, 0]):
            The number of group tokens for each stage.
        num_output_groups (`list[int]`, *optional*, defaults to [64, 8, 8]):
            The number of output groups for each stage, 0 means no group.
        num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import GroupViTVisionConfig, GroupViTVisionModel

    >>> # Initializing a GroupViTVisionModel with nvidia/groupvit-gcc-yfcc style configuration
    >>> configuration = GroupViTVisionConfig()

    >>> model = GroupViTVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "groupvit_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    depths: list[int] | tuple[int, ...] = (6, 3, 3)
    num_group_tokens: list[int] | tuple[int, ...] = (64, 8, 0)
    num_output_groups: list[int] | tuple[int, ...] = (64, 8, 8)
    num_attention_heads: int = 6
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    assign_eps: float = 1.0
    assign_mlp_ratio: list[float | int] | tuple[float | int, ...] = (0.5, 4)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_hidden_layers != sum(self.depths):
            logger.warning(
                f"Manually setting num_hidden_layers to {self.num_hidden_layers}, but we expect num_hidden_layers ="
                f" sum(depth) = {sum(self.depths)}"
            )


@strict(accept_kwargs=True)
@dataclass(repr=False)
class GroupViTConfig(PreTrainedConfig):
    r"""
    [`GroupViTConfig`] is the configuration class to store the configuration of a [`GroupViTModel`]. It is used to
    instantiate a GroupViT model according to the specified arguments, defining the text model and vision model
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the GroupViT
    [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`GroupViTTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`GroupViTVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 256):
            Dimensionality of text and vision projection layers.
        projection_intermediate_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of intermediate layer of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original GroupViT
            implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "groupvit"
    sub_configs = {"text_config": GroupViTTextConfig, "vision_config": GroupViTVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 256
    projection_intermediate_dim: int = 4096
    logit_scale_init_value: float = 2.6592
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    output_segmentation: bool = False

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `GroupViTTextConfig` with default values.")
        elif isinstance(self.text_config, GroupViTTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `GroupViTVisionConfig` with default values.")
        elif isinstance(self.vision_config, GroupViTVisionConfig):
            vision_config = self.vision_config.to_dict()
        else:
            vision_config = self.vision_config

        # For backward compatibility check keyword args
        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        if text_config_dict is not None:
            # This is the complete result when using `text_config_dict`.
            _text_config_dict = GroupViTTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key != "transformers_version":
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `GroupViTTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = GroupViTVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key != "transformers_version":
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `GroupViTVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        # Finally we can convert back our unified text/vision configs to `PretrainedConfig`
        self.text_config = GroupViTTextConfig(**text_config)
        self.vision_config = GroupViTVisionConfig(**vision_config)

        super().__post_init__(**kwargs)


__all__ = ["GroupViTConfig", "GroupViTTextConfig", "GroupViTVisionConfig"]
