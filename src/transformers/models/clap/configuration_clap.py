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
"""CLAP model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ClapTextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapTextModel`]. It is used to instantiate a CLAP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLAP
    [calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the CLAP model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ClapTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`,
            `"relu"`, `"silu"` and `"relu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`ClapTextModel`].
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        projection_dim (`int`, *optional*, defaults to 512)
            Dimension of the projection head of the `ClapTextModelWithProjection`.

    Examples:

    ```python
    >>> from transformers import ClapTextConfig, ClapTextModel

    >>> # Initializing a CLAP text configuration
    >>> configuration = ClapTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ClapTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clap_text_model"
    base_config_key = "text_config"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_factor: float = 1.0
    layer_norm_eps: float = 1e-12
    projection_dim: int = 512
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    projection_hidden_act: str = "relu"


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ClapAudioConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapAudioModel`]. It is used to instantiate a
    CLAP audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        window_size (`int`, *optional*, defaults to 8):
            Image size of the spectrogram
        num_mel_bins (`int`, *optional*, defaults to 64):
            Number of mel features used per frames. Should correspond to the value used in the `ClapProcessor` class.
        spec_size (`int`, *optional*, defaults to 256):
            Desired input size of the spectrogram that the model supports. It can be different from the output of the
            `ClapFeatureExtractor`, in which case the input features will be resized. Corresponds to the `image_size`
            of the audio models.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        patch_size (`int`, *optional*, defaults to 4):
            Patch size for the audio spectrogram
        patch_stride (`list`, *optional*, defaults to `[4, 4]`):
            Patch stride for the audio spectrogram
        num_classes (`int`, *optional*, defaults to 527):
            Number of classes used for the head training
        hidden_size (`int`, *optional*, defaults to 768):
            Hidden size of the output of the audio encoder. Correspond to the dimension of the penultimate layer's
            output,which is sent to the projection MLP layer.
        projection_dim (`int`, *optional*, defaults to 512):
            Hidden size of the projection layer.
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            Depths used for the Swin Layers of the audio model
        num_attention_heads (`list`, *optional*, defaults to `[4, 8, 16, 32]`):
            Number of attention heads used for the Swin Layers of the audio model
        enable_fusion (`bool`, *optional*, defaults to `False`):
            Whether or not to enable patch fusion. This is the main contribution of the authors, and should give the
            best results.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder.
        fusion_type (`str`, *optional*):
            Fusion type used for the patch fusion.
        patch_embed_input_channels (`int`, *optional*, defaults to 1):
            Number of channels used for the input spectrogram
        flatten_patch_embeds (`bool`, *optional*, defaults to `True`):
            Whether or not to flatten the patch embeddings
        patch_embeds_hidden_size (`int`, *optional*, defaults to 96):
            Hidden size of the patch embeddings. It is used as the number of output channels.
        enable_patch_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether or not to enable layer normalization for the patch embeddings
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Drop path rate for the patch fusion
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to add a bias to the query, key, value projections.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the mlp hidden dim to embedding dim.
        aff_block_r (`int`, *optional*, defaults to 4):
            downsize_ratio used in the AudioFF block
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        layer_norm_eps (`[type]`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import ClapAudioConfig, ClapAudioModel

    >>> # Initializing a ClapAudioConfig with laion/clap-htsat-fused style configuration
    >>> configuration = ClapAudioConfig()

    >>> # Initializing a ClapAudioModel (with random weights) from the laion/clap-htsat-fused style configuration
    >>> model = ClapAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clap_audio_model"
    base_config_key = "audio_config"

    window_size: int = 8
    num_mel_bins: int = 64
    spec_size: int = 256
    hidden_act: str = "gelu"
    patch_size: int | list[int] | tuple[int, int] = 4
    patch_stride: int | list[int] | tuple[int, ...] = (4, 4)
    num_classes: int = 527
    hidden_size: int = 768
    projection_dim: int = 512
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    enable_fusion: bool = False
    hidden_dropout_prob: float = 0.1
    fusion_type: str | None = None
    patch_embed_input_channels: int = 1
    flatten_patch_embeds: bool = True
    patch_embeds_hidden_size: int = 96
    enable_patch_layer_norm: bool = True
    drop_path_rate: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    qkv_bias: bool = True
    mlp_ratio: float = 4.0
    aff_block_r: int = 4
    num_hidden_layers: int = 4
    projection_hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5
    initializer_factor: float = 1.0


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ClapConfig(PreTrainedConfig):
    r"""
    [`ClapConfig`] is the configuration class to store the configuration of a [`ClapModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapAudioConfig`].
        logit_scale_init_value (`float`, *optional*, defaults to 14.29):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLAP implementation.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and audio projection layers.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function for the projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClapConfig, ClapModel

    >>> # Initializing a ClapConfig with laion-ai/base style configuration
    >>> configuration = ClapConfig()

    >>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
    >>> model = ClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
    >>> from transformers import ClapTextConfig, ClapAudioConfig

    >>> # Initializing a ClapText and ClapAudioConfig configuration
    >>> config_text = ClapTextConfig()
    >>> config_audio = ClapAudioConfig()

    >>> config = ClapConfig(text_config=config_text, audio_config=config_audio)
    ```"""

    model_type = "clap"
    sub_configs = {"text_config": ClapTextConfig, "audio_config": ClapAudioConfig}

    text_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    logit_scale_init_value: float = 1 / 0.07
    projection_dim: int = 512
    projection_hidden_act: str = "relu"
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = ClapTextConfig()
            logger.info("`text_config` is `None`. initializing the `ClapTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = ClapTextConfig(**self.text_config)

        if self.audio_config is None:
            self.audio_config = ClapAudioConfig()
            logger.info("`audio_config` is `None`. initializing the `ClapAudioConfig` with default values.")
        elif isinstance(self.audio_config, dict):
            self.audio_config = ClapAudioConfig(**self.audio_config)

        self.text_config.projection_dim = self.projection_dim
        self.audio_config.projection_dim = self.projection_dim

        self.text_config.projection_hidden_act = self.projection_hidden_act
        self.audio_config.projection_hidden_act = self.projection_hidden_act
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers + len(self.audio_config.depths)
        super().__post_init__(**kwargs)


__all__ = ["ClapAudioConfig", "ClapConfig", "ClapTextConfig"]
