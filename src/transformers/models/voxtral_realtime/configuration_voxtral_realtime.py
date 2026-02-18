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

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from ..auto import CONFIG_MAPPING, AutoConfig
from ..mistral.configuration_mistral import MistralConfig


class VoxtralRealtimeTextConfig(MistralConfig):
    r"""
    This is the configuration class to store the configuration of a [`VoxtralRealtimeText`]. It is used to instantiate a
    Voxtral Realtime text decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text decoder of the Voxtral Realtime
    architecture.

    e.g. [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
    """

    model_type = "voxtral_realtime_text"


@strict(accept_kwargs=True)
@dataclass(repr=False)
class VoxtralRealtimeEncoderConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`VoxtralRealtimeEncoder`]. It is used to instantiate a
    Voxtral Realtime audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Voxtral Realtime
    architecture.

    e.g. [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
            vocab_size (`int`, *optional*, defaults to 131072):
                Vocabulary size of the model.
            hidden_size (`int`, *optional*, defaults to 1280):
                Dimensionality of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 5120):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer encoder.
            activation_function (`str`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the encoder and pooler.
            num_mel_bins (`int`, *optional*, defaults to 128):
                Number of mel features used per input features. Should correspond to the value used in the
                `VoxtralRealtimeProcessor` class.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            hidden_act (`str`, *optional*, defaults to `"silu"`):
                The activation function used in the MLP layers.
            max_position_embeddings (`int`, *optional*, defaults to 1500):
                The maximum sequence length that this model might ever be used with.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                The epsilon used by the RMS normalization layers.
            rope_parameters (`Union[RopeParameters, dict]`, *optional*):
                The parameters for the rotary position embeddings.
            sliding_window (`int`, *optional*, defaults to 750):
                The sliding window size for local attention.
            head_dim (`int`, *optional*, defaults to 64):
                The dimension of each attention head.

    ```python
    >>> from transformers import VoxtralRealtimeEncoderConfig, VoxtralRealtimeEncoder

    >>> # Initializing a VoxtralRealtimeEncoderConfig
    >>> configuration = VoxtralRealtimeEncoderConfig()

    >>> # Initializing a VoxtralRealtimeEncoder (with random weights)
    >>> model = VoxtralRealtimeEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "voxtral_realtime_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
        "num_key_value_heads": "num_attention_heads",
    }

    vocab_size: int = 131072
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    activation_function: str = "gelu"
    num_mel_bins: int = 128
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    hidden_act: str = "silu"
    max_position_embeddings: int = 1500
    rms_norm_eps: float = 1e-05
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int = 750
    head_dim: int = 64

    def __post_init__(self, **kwargs):
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class VoxtralRealtimeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VoxtralRealtimeForConditionalGeneration`]. It is used to instantiate a
    Voxtral Realtime model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Voxtral Realtime.

    e.g. [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio encoder.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function (function or string) in the multi-modal projector.
        audio_length_per_tok (`int`, *optional*, defaults to 8):
            The number of audio frames corresponding to each text token.
        default_num_delay_tokens (`int`, *optional*, defaults to 6):
            The default number of delay tokens used for streaming.
        downsample_factor (`int`, *optional*, defaults to 4):
            The downsampling factor applied to audio features before projection.

    ```python
    >>> from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeConfig

    >>> # Initializing a VoxtralRealtime configuration
    >>> configuration = VoxtralRealtimeConfig()

    >>> # Initializing a model with random weights
    >>> model = VoxtralRealtimeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "voxtral_realtime"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 131072,
        "hidden_size": 3072,
        "intermediate_size": 9216,
        "num_hidden_layers": 26,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "rope_theta": 1000000.0,
        "head_dim": 128,
        "tie_word_embeddings": True,
        "sliding_window": 8192,
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    projector_hidden_act: str = "gelu"
    audio_length_per_tok: int = 8
    default_num_delay_tokens: int = 6
    downsample_factor: int = 4

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "voxtral_realtime_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["voxtral_realtime_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "voxtral_realtime_text")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["voxtral_realtime_text"](**self._default_text_config_kwargs)

        self.hidden_size = self.text_config.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["VoxtralRealtimeEncoderConfig", "VoxtralRealtimeConfig", "VoxtralRealtimeTextConfig"]
