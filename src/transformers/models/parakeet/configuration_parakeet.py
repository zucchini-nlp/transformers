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
"""Parakeet model configuration."""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ParakeetEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParakeetEncoder`]. It is used to instantiate a
    `ParakeetEncoder` model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the layers and the hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the attention layers.
        convolution_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in convolutions of the conformer's convolution module.
        conv_kernel_size (`int`, *optional*, defaults to 9):
            The kernel size of the convolution layers in the Conformer block.
        subsampling_factor (`int`, *optional*, defaults to 8):
            The factor by which the input sequence is subsampled.
        subsampling_conv_channels (`int`, *optional*, defaults to 256):
            The number of channels in the subsampling convolution layers.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel features.
        subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size of the subsampling convolution layers.
        subsampling_conv_stride (`int`, *optional*, defaults to 2):
            The stride of the subsampling convolution layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for all fully connected layers in the embeddings, encoder, and pooler.
        dropout_positions (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the positions in the input sequence.
        layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the layers in the encoder.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention layers.
        max_position_embeddings (`int`, *optional*, defaults to 5000):
            The maximum sequence length that this model might ever be used with.
        scale_input (`bool`, *optional*, defaults to `True`):
            Whether to scale the input embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
        ```python
        >>> from transformers import ParakeetEncoderModel, ParakeetEncoderConfig

        >>> # Initializing a `ParakeetEncoder` configuration
        >>> configuration = ParakeetEncoderConfig()

        >>> # Initializing a model from the configuration
        >>> model = ParakeetEncoderModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the ParakeetEncoder architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "parakeet_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = True
    convolution_bias: bool = True
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 80
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    dropout: float = 0.1
    dropout_positions: float = 0.0
    layerdrop: float = 0.1
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 5000
    scale_input: bool = True
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ParakeetCTCConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParakeetForCTC`]. It is used to instantiate a
    Parakeet CTC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
            vocab_size (`int`, *optional*, defaults to 1025):
                Vocabulary size of the model.
            ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
                Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
                instance of [`ParakeetForCTC`].
            ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
                Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
                occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
                of [`ParakeetForCTC`].
            encoder_config (`Union[dict, ParakeetEncoderConfig]`, *optional*):
                The config object or dictionary of the encoder.
            pad_token_id (`int`, *optional*, defaults to 1024):
                Padding token id. Also used as blank token id.

    Example:
        ```python
        >>> from transformers import ParakeetForCTC, ParakeetCTCConfig

        >>> # Initializing a Parakeet configuration
        >>> configuration = ParakeetCTCConfig()

        >>> # Initializing a model from the configuration
        >>> model = ParakeetForCTC(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the Parakeet CTC architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "parakeet_ctc"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    vocab_size: int = 1025
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int | None = 1024

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = ParakeetEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


__all__ = ["ParakeetCTCConfig", "ParakeetEncoderConfig"]
