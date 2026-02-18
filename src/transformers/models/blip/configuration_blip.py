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
"""Blip model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BlipTextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipTextModel`]. It is used to instantiate a BLIP
    text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the `BlipText` used by the [base
    architectures](https://huggingface.co/Salesforce/blip-vqa-base).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30524):
            Vocabulary size of the `Blip` text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`BlipModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        encoder_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers from the vision model.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        bos_token_id (`int`, *optional*, defaults to 30522):
            The id of the `beginning-of-sequence` token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the `end-of-sequence` token.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the `padding` token.
        sep_token_id (`int`, *optional*, defaults to 102):
            The id of the `separator` token.
        is_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as a decoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        label_smoothing (float, *optional*):
            A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://huggingface.co/papers/1512.00567>`__. Default: :math:`0.0`.

    Example:

    ```python
    >>> from transformers import BlipTextConfig, BlipTextModel

    >>> # Initializing a BlipTextConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipTextConfig()

    >>> # Initializing a BlipTextModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30524
    hidden_size: int = 768
    encoder_hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 512
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    bos_token_id: int | None = 30522
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = 0
    sep_token_id: int | None = 102
    is_decoder: bool = True
    use_cache: bool = True
    label_smoothing: float = 0.0


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BlipVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipVisionModel`]. It is used to instantiate a
    BLIP vision model according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the Blip-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import BlipVisionConfig, BlipVisionModel

    >>> # Initializing a BlipVisionConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipVisionConfig()

    >>> # Initializing a BlipVisionModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    image_size: int = 384
    patch_size: int = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BlipConfig(PreTrainedConfig):
    r"""
    [`BlipConfig`] is the configuration class to store the configuration of a [`BlipModel`]. It is used to instantiate
    a BLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the BLIP-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original BLIP implementation.
        image_text_hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden state of the image-text fusion layer.
        label_smoothing (float, optional, *optional*, defaults to 0.0):
            A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://huggingface.co/papers/1512.00567>`__. Default: :math:`0.0`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings

    Example:

    ```python
    >>> from transformers import BlipConfig, BlipModel

    >>> # Initializing a BlipConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipConfig()

    >>> # Initializing a BlipPModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a BlipConfig from a BlipTextConfig and a BlipVisionConfig

    >>> # Initializing a BLIPText and BLIPVision configuration
    >>> config_text = BlipTextConfig()
    >>> config_vision = BlipVisionConfig()

    >>> config = BlipConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "blip"
    sub_configs = {"text_config": BlipTextConfig, "vision_config": BlipVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    image_text_hidden_size: int = 256
    label_smoothing: float = 0.0
    tie_word_embeddings: bool = True
    initializer_factor: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = BlipTextConfig()
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = BlipTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = BlipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `BlipVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = BlipVisionConfig(**self.vision_config)

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        super().__post_init__(**kwargs)


__all__ = ["BlipConfig", "BlipTextConfig", "BlipVisionConfig"]
