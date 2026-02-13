# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""Musicgen Melody model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class MusicgenMelodyDecoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MusicgenMelodyDecoder`]. It is used to instantiate a
    Musicgen Melody decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the MusicgenMelodyDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MusicgenMelodyDecoder`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(hidden_size).
        num_codebooks (`int`, *optional*, defaults to 4):
            The number of parallel codebooks forwarded to the model.
        audio_channels (`int`, *optional*, defaults to 1):
            Number of audio channels used by the model (either mono or stereo). Stereo models generate a separate
            audio stream for the left/right output channels. Mono models generate a single audio stream output.
        pad_token_id (`int`, *optional*, defaults to 2048): The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 2048): The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*): The id of the *end-of-sequence* token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`): Whether to tie word embeddings with the text encoder.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on
            decoder-only or encoder-only architectures.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model.
    """

    model_type = "musicgen_melody_decoder"
    base_config_key = "decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 2048
    max_position_embeddings: int = 2048
    num_hidden_layers: int = 24
    ffn_dim: int = 4096
    num_attention_heads: int = 16
    layerdrop: float = 0.0
    use_cache: bool = True
    activation_function: str = "gelu"
    hidden_size: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    initializer_factor: float = 0.02
    scale_embedding: bool = False
    num_codebooks: int = 4
    audio_channels: int = 1
    pad_token_id: int | None = 2048
    bos_token_id: int | None = 2048
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    is_decoder: bool = False
    add_cross_attention: bool = False

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.audio_channels not in [1, 2]:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) audio channels, got {self.audio_channels} channels.")


@strict(accept_kwargs=True)
@dataclass(repr=False)
class MusicgenMelodyConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicgenMelodyModel`]. It is used to instantiate a
    Musicgen Melody model according to the specified arguments, defining the text encoder, audio encoder and Musicgen Melody decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_encoder (`Union[dict, `PretrainedConfig`]`):
            An instance of a configuration object that defines the text encoder config.
        audio_encoder (`Union[dict, `PretrainedConfig`]`):
            An instance of a configuration object that defines the audio encoder config.
        decoder (`Union[dict, `PretrainedConfig`]`):
            An instance of a configuration object that defines the decoder config.
        num_chroma (`int`, *optional*, defaults to 12):
            Number of chroma bins to use.
        chroma_length (`int`, *optional*, defaults to 235):
            Maximum chroma duration if audio is used to condition the model. Corresponds to the maximum duration used during training.

    Example:

    ```python
    >>> from transformers import (
    ...     MusicgenMelodyConfig,
    ...     MusicgenMelodyDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenMelodyForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenMelodyDecoderConfig()

    >>> configuration = MusicgenMelodyConfig(
    ...     text_encoder=text_encoder_config, audio_encoder=audio_encoder_config, decoder=decoder_config
    ... )

    >>> # Initializing a MusicgenMelodyForConditionalGeneration (with random weights) from the facebook/musicgen-melody style configuration
    >>> model = MusicgenMelodyForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen_melody-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_melody_config = MusicgenMelodyConfig.from_pretrained("musicgen_melody-model")
    >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("musicgen_melody-model", config=musicgen_melody_config)
    ```"""

    model_type = "musicgen_melody"
    sub_configs = {
        "text_encoder": AutoConfig,
        "audio_encoder": AutoConfig,
        "decoder": MusicgenMelodyDecoderConfig,
    }
    has_no_defaults_at_init = True

    text_encoder: dict | PreTrainedConfig = None
    audio_encoder: dict | PreTrainedConfig = None
    decoder: dict | PreTrainedConfig = None
    tie_encoder_decoder: bool = False
    num_chroma: int = 12
    chroma_length: int = 235
    initializer_factor: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.text_encoder, dict):
            text_encoder_model_type = self.text_encoder.pop("model_type")
            self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **self.text_encoder)

        if isinstance(self.audio_encoder, dict):
            audio_encoder_model_type = self.audio_encoder.pop("model_type")
            self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **self.audio_encoder)

        if isinstance(self.decoder, dict):
            self.decoder = MusicgenMelodyDecoderConfig(**self.decoder)

        self.is_encoder_decoder = True
        super().__post_init__(**kwargs)

    @property
    # This is a property because you might want to change the codec model on the fly
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate


__all__ = ["MusicgenMelodyConfig", "MusicgenMelodyDecoderConfig"]
