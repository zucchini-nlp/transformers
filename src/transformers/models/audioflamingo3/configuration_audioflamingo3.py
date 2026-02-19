# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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
from ..auto import CONFIG_MAPPING, AutoConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class AudioFlamingo3EncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AudioFlamingo3Encoder`]. It is used to instantiate an
    AudioFlamingo3 audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the AudioFlamingo3
    architecture.

    e.g. [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `AudioFlamingo3Processor` class.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556)
            for more details.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by dividing by sqrt(hidden_size).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3EncoderConfig, AudioFlamingo3Encoder

    >>> # Initializing an AudioFlamingo3EncoderConfig
    >>> configuration = AudioFlamingo3EncoderConfig()

    >>> # Initializing an AudioFlamingo3Encoder (with random weights)
    >>> model = AudioFlamingo3Encoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audioflamingo3_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    num_mel_bins: int = 128
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    layerdrop: float | int = 0.0
    activation_function: str = "gelu"
    hidden_size: int = 1280
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    initializer_range: float = 0.02
    scale_embedding: bool = False
    max_source_positions: int = 1500


@strict(accept_kwargs=True)
@dataclass(repr=False)
class AudioFlamingo3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AudioFlamingo3ForConditionalGeneration`]. It is used to instantiate an
    AudioFlamingo3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AudioFlamingo3.

    e.g. [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AudioFlamingo3EncoderConfig, dict]`, *optional*, defaults to `AudioFlamingo3EncoderConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151669):
            The audio token index to encode the audio prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function used in the projector.
        projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to include bias terms in the projector.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Config, AudioFlamingo3EncoderConfig, Qwen2Config

    >>> # Initializing an AudioFlamingo3Encoder config
    >>> audio_config = AudioFlamingo3EncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing an AudioFlamingo3 configuration
    >>> configuration = AudioFlamingo3Config(audio_config, text_config)

    >>> # Initializing a model from the audioflamingo3 style configuration
    >>> model = AudioFlamingo3ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audioflamingo3"
    sub_configs = {
        "audio_config": AudioFlamingo3EncoderConfig,
        "text_config": AutoConfig,
    }
    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151669
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "audioflamingo3_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["audioflamingo3_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["AudioFlamingo3Config", "AudioFlamingo3EncoderConfig"]
