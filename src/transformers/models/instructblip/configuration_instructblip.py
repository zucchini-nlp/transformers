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
"""InstructBLIP model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class InstructBlipVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InstructBlipVisionModel`]. It is used to
    instantiate a InstructBLIP vision encoder according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the InstructBLIP
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 39):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported. to 1e-5): The epsilon used by the layer
            normalization layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries and values in the self-attention layers.

    Example:

    ```python
    >>> from transformers import InstructBlipVisionConfig, InstructBlipVisionModel

    >>> # Initializing a InstructBlipVisionConfig with Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipVisionConfig()

    >>> # Initializing a InstructBlipVisionModel (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "instructblip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1408
    intermediate_size: int = 6144
    num_hidden_layers: int = 39
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True


@strict(accept_kwargs=True)
@dataclass(repr=False)
class InstructBlipQFormerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InstructBlipQFormerModel`]. It is used to
    instantiate a InstructBLIP Querying Transformer (Q-Former) model according to the specified arguments, defining the
    model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the InstructBLIP [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5)
    architecture. Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PreTrainedConfig`] for more information.

    Note that [`InstructBlipQFormerModel`] is very similar to [`BertLMHeadModel`] with interleaved cross-attention.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling the model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Token id used for padding sequences.
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            The frequency of adding cross-attention to the Transformer layers.
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            The hidden size of the hidden states for cross-attention.

    Examples:

    ```python
    >>> from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel

    >>> # Initializing a InstructBLIP Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipQFormerConfig()

    >>> # Initializing a model (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipQFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "instructblip_qformer"
    base_config_key = "qformer_config"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    cross_attention_frequency: int = 2
    encoder_hidden_size: int = 1408


@strict(accept_kwargs=True)
@dataclass(repr=False)
class InstructBlipConfig(PreTrainedConfig):
    r"""
    [`InstructBlipConfig`] is the configuration class to store the configuration of a
    [`InstructBlipForConditionalGeneration`]. It is used to instantiate a InstructBLIP model according to the specified
    arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the InstructBLIP
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipVisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipQFormerConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PreTrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        image_token_index (`int`, *optional*):
            Token index of special image token.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     InstructBlipVisionConfig,
    ...     InstructBlipQFormerConfig,
    ...     OPTConfig,
    ...     InstructBlipConfig,
    ...     InstructBlipForConditionalGeneration,
    ... )

    >>> # Initializing a InstructBlipConfig with Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipConfig()

    >>> # Initializing a InstructBlipForConditionalGeneration (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a InstructBlipConfig from a InstructBlipVisionConfig, InstructBlipQFormerConfig and any PreTrainedConfig

    >>> # Initializing InstructBLIP vision, InstructBLIP Q-Former and language model configurations
    >>> vision_config = InstructBlipVisionConfig()
    >>> qformer_config = InstructBlipQFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = InstructBlipConfig(vision_config=vision_config, qformer_config=qformer_config, text_config=text_config)
    ```"""

    model_type = "instructblip"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "qformer_config": InstructBlipQFormerConfig,
        "vision_config": InstructBlipVisionConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    qformer_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    num_query_tokens: int = 32
    image_token_index: int | None = None
    initializer_factor: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = CONFIG_MAPPING["opt"]()
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")
        elif isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type", "opt")
            self.text_config = CONFIG_MAPPING[text_model_type](**self.text_config)

        if self.qformer_config is None:
            self.qformer_config = InstructBlipQFormerConfig()
            logger.info("qformer_config is None. Initializing the InstructBlipQFormerConfig with default values.")
        elif isinstance(self.qformer_config, dict):
            self.qformer_config = InstructBlipQFormerConfig(**self.qformer_config)

        if self.vision_config is None:
            self.vision_config = InstructBlipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `InstructBlipVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = InstructBlipVisionConfig(**self.vision_config)

        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        super().__post_init__(**kwargs)


__all__ = ["InstructBlipConfig", "InstructBlipQFormerConfig", "InstructBlipVisionConfig"]
