# Copyright 2024 Descript and The HuggingFace Inc. team. All rights reserved.
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
"""Dac model configuration"""

import math
from dataclasses import dataclass

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class DacConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`DacModel`]. It is used to instantiate a
    Dac model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [descript/dac_16khz](https://huggingface.co/descript/dac_16khz) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        encoder_hidden_size (`int`, *optional*, defaults to 64):
            Intermediate representation dimension for the encoder.
        downsampling_ratios (`list[int]`, *optional*, defaults to `[2, 4, 8, 8]`):
            Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 1536):
            Intermediate representation dimension for the decoder.
        n_codebooks (`int`, *optional*, defaults to 9):
            Number of codebooks in the VQVAE.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of discrete codes in each codebook.
        codebook_dim (`int`, *optional*, defaults to 8):
            Dimension of the codebook vectors. If not defined, uses `encoder_hidden_size`.
        quantizer_dropout (`bool`, *optional*, defaults to 0):
            Whether to apply dropout to the quantizer.
        commitment_loss_weight (float, *optional*, defaults to 0.25):
            Weight of the commitment loss term in the VQVAE loss function.
        codebook_loss_weight (float, *optional*, defaults to 1.0):
            Weight of the codebook loss term in the VQVAE loss function.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
    Example:

    ```python
    >>> from transformers import DacModel, DacConfig

    >>> # Initializing a "descript/dac_16khz" style configuration
    >>> configuration = DacConfig()

    >>> # Initializing a model (with random weights) from the "descript/dac_16khz" style configuration
    >>> model = DacModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dac"

    encoder_hidden_size: int = 64
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 4, 8, 8)
    decoder_hidden_size: int = 1536
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: int = 8
    quantizer_dropout: float = 0.0
    commitment_loss_weight: float = 0.25
    codebook_loss_weight: float = 1.0
    sampling_rate: int = 16000

    def __post_init__(self, **kwargs):
        self.upsampling_ratios = self.downsampling_ratios[::-1]
        self.hidden_size = self.encoder_hidden_size * (2 ** len(self.downsampling_ratios))
        self.hop_length = int(np.prod(self.downsampling_ratios))
        super().__post_init__(**kwargs)

    @property
    def frame_rate(self) -> int:
        hop_length = np.prod(self.upsampling_ratios)
        return math.ceil(self.sampling_rate / hop_length)


__all__ = ["DacConfig"]
