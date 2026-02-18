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
"""CLVP model configuration"""

import os
from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ClvpEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClvpEncoder`]. It is used to instantiate a CLVP
    text or CLVP speech encoder according to the specified arguments. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the CLVP Encoder model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the projection vector.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the feed-forward layers in [`ClvpEncoderMLP`].
        use_rotary_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use rotary_embedding or not.
        use_attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Query, Key and Value layers during self attention.
        summary_type (`str`, *optional*, defaults to `"mean"`):
            What strategy to use to get pooler_output from the last_hidden_state. `"last"`, `"first"`, `"mean"` and
            `"cls_index"` are supported.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        bos_token_id (`int`, *optional*, defaults to 255):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 0):
            End of sequence token id.
        pad_token_id (`int`, *optional*):
            Padding token id.

    Example:

    ```python
    >>> from transformers import ClvpEncoderConfig, ClvpEncoder

    >>> # Initializing a ClvpEncoderConfig with susnato/clvp_dev style configuration
    >>> encoder_configuration = ClvpEncoderConfig()

    >>> # Initializing a ClvpEncoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpEncoder(encoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clvp_encoder"
    base_config_key = ["text_config", "speech_config"]

    vocab_size: int = 256
    hidden_size: int = 768
    intermediate_size: int = 1536
    projection_dim: int = 768
    num_hidden_layers: int = 20
    num_attention_heads: int = 12
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.1
    dropout: float | int = 0.1
    use_rotary_embedding: bool = True
    use_attention_bias: bool = False
    summary_type: str = "mean"
    initializer_factor: float = 1.0
    bos_token_id: int | None = 255
    eos_token_id: int | list[int] | None = 0
    pad_token_id: int | None = None

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, config_type: str = "text_config", **kwargs
    ):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # make sure to have the config_type be either "text_config" or "speech_config"
        # this is to make sure that we can load only text or speech configs from the nested ClvpConfig.
        if config_type not in cls.base_config_key:
            raise ValueError(
                f"We can only load either 'text_config' or 'speech_config' but you are trying to load{config_type}"
            )

        # get the text config dict if we are loading from ClvpConfig
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict[config_type]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ClvpDecoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClvpDecoder`]. It is used to instantiate a CLVP
    Decoder Model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Decoder part of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    The architecture is similar to GPT2.

    Args:
        vocab_size (`int`, *optional*, defaults to 8194):
            Vocabulary size of the model.
        max_position_embeddings (`int`, *optional*, defaults to 608):
            The maximum sequence length of mel tokens that this model might ever be used with. Similar to `n_positions`
            in `GPT2Config`.
        max_text_tokens (`int`, *optional*, defaults to 404):
            The maximum sequence length of text tokens that this model might ever be used with. Similar to
            `n_positions` in `GPT2Config`.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 30):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times `hidden_size`.
        num_mel_attn_blocks (`int`, *optional*, defaults to 6):
            Denotes the number of self attention layers in [`ClvpConditioningEncoder`].
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary.

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio to be used after the projection and activation.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 8192):
            Beginning of sequence token id, used at the start of the generation.
        eos_token_id (`int`, *optional*, defaults to 8193):
            End of sequence token id, used in the method
            [`ClvpModelForConditionalGeneration.fix_speech_decoder_output()`] to correct decoder outputs.
        pad_token_id (`int`, *optional*):
            Padding token id.
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted mel features. This value is used in [`ClvpConditioningEncoder`].
        use_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in Query, Key and Value layers during self attention.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        decoder_fixing_codes (`list`, *optional*, defaults to `[83, 45, 45, 248]`):
            These values are used in the method `fix_speech_decoder_output` to fix decoder generated outputs.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model.

    Example:

    ```python
    >>> from transformers import ClvpDecoderConfig, ClvpDecoder

    >>> # Initializing a ClvpDecoderConfig with susnato/clvp_dev style configuration
    >>> decoder_configuration = ClvpDecoderConfig()

    >>> # Initializing a ClvpDecoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpDecoder(decoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clvp_decoder"
    base_config_key = "decoder_config"

    vocab_size: int = 8194
    max_position_embeddings: int = 608
    max_text_tokens: int = 404
    hidden_size: int = 1024
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    n_inner: int | None = None
    num_mel_attn_blocks: int = 6
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attention_dropout: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: str | None = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float | int = 0.1
    use_cache: bool = True
    bos_token_id: int | None = 8192
    eos_token_id: int | None = 8193
    pad_token_id: int | None = None
    feature_size: int = 80
    use_attention_bias: bool = True
    initializer_factor: float = 1.0
    decoder_fixing_codes: list[int] | tuple[int, ...] = (83, 45, 45, 248)
    add_cross_attention: bool = False


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ClvpConfig(PreTrainedConfig):
    r"""
    [`ClvpConfig`] is the configuration class to store the configuration of a [`ClvpModelForConditionalGeneration`]. It
    is used to instantiate a CLVP model according to the specified arguments, defining the text model, speech model and
    decoder model configs. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the CLVP [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the CLVP text encoder.
        speech_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize CLVP speech encoder.
        decoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClvpDecoderConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of text and speech projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLVP implementation.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClvpConfig, ClvpModelForConditionalGeneration

    >>> # Initializing a ClvpConfig with susnato/clvp_dev style configuration
    >>> configuration = ClvpConfig()

    >>> # Initializing a ClvpModelForConditionalGeneration (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpModelForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLVPConfig from a CLVPTextConfig, CLVPSpeechConfig and a CLVPAutoRegressiveConfig
    >>> from transformers import ClvpEncoderConfig, ClvpDecoderConfig

    >>> # Initializing a CLVP text, CLVP speech and CLVP decoder configuration
    >>> config_text = ClvpEncoderConfig()
    >>> config_speech = ClvpEncoderConfig()
    >>> decoder_config = ClvpDecoderConfig()

    >>> config = ClvpConfig(config_text, config_speech, decoder_config)
    ```"""

    model_type = "clvp"
    sub_configs = {
        "text_config": ClvpEncoderConfig,
        "speech_config": ClvpEncoderConfig,
        "decoder_config": ClvpDecoderConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    speech_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 768
    logit_scale_init_value: float = 2.6592
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = ClvpEncoderConfig()
            logger.info("`text_config` is `None`. initializing the `ClvpEncoderConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = ClvpEncoderConfig(**self.text_config)

        if self.speech_config is None:
            self.speech_config = ClvpEncoderConfig()
            logger.info("`speech_config` is `None`. initializing the `ClvpEncoderConfig` with default values.")
        elif isinstance(self.speech_config, dict):
            self.speech_config = ClvpEncoderConfig(**self.speech_config)

        if self.decoder_config is None:
            self.decoder_config = ClvpDecoderConfig()
            logger.info("`image_config` is `None`. initializing the `ClvpDecoderConfig` with default values.")
        elif isinstance(self.decoder_config, dict):
            self.decoder_config = ClvpDecoderConfig(**self.decoder_config)

        super().__post_init__(**kwargs)


__all__ = ["ClvpConfig", "ClvpDecoderConfig", "ClvpEncoderConfig"]
