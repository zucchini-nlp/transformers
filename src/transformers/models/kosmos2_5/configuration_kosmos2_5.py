# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""KOSMOS-2.5 model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Kosmos2_5TextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2_5TextModel`]. It is used to instantiate a
    KOSMOS-2.5 text decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text decoder of the KOSMOS-2.5
    [microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 108481):
            Vocabulary size of the Kosmos2_5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Kosmos2_5Model`].
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the layers and the pooler layer.
        layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        ffn_dim (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see
            https://huggingface.co/papers/1909.11556) for more details.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(embed_dim).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    ```"""

    model_type = "kosmos_2_5_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    vocab_size: int = 108481
    max_position_embeddings: int = 4096
    embed_dim: int = 1536
    layers: int = 24
    ffn_dim: int = 6144
    attention_heads: int = 16
    activation_function: str = "gelu"
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    layerdrop: float = 0.0
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    scale_embedding: bool = True
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Kosmos2_5VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2_5VisionModel`]. It is used to
    instantiate a KOSMOS-2.5 vision encoder according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the vision encoder of the KOSMOS-2.5
    [microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        patch_embed_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the input patch_embedding layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3968):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of the key, query, value projections per attention head.
        num_hidden_layers (`int`, *optional*, defaults to 18):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 24):
            Number of attention heads for each attention layer in the Transformer encoder.
        dense_act_fn (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_num_patches (`int`, *optional*, defaults to 4096):
            Maximum sequence length (here number of patches) supported by the model.
    Example:

    ```python
    >>> from transformers import Kosmos2_5VisionConfig, Kosmos2_5VisionModel

    >>> # Initializing a Kosmos2_5VisionConfig with microsoft/kosmos-2.5 style configuration
    >>> configuration = Kosmos2_5VisionConfig()

    >>> # Initializing a Kosmos2_5VisionModel (with random weights) from the microsoft/kosmos-2.5 style configuration
    >>> model = Kosmos2_5VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kosmos_2_5_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1536
    patch_embed_hidden_size: int = 768
    intermediate_size: int = 3968
    head_dim: int = 64
    num_hidden_layers: int = 18
    num_attention_heads: int = 24
    dense_act_fn: str = "gelu_new"
    layer_norm_eps: float = 1e-6
    dropout_rate: float = 0.0
    attention_dropout: float | int = 0.0
    max_num_patches: int = 4096
    initializer_factor: float = 1.0
    initializer_range: float = 0.02


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Kosmos2_5Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2_5Model`]. It is used to instantiate a
    KOSMOS-2.5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the KOSMOS-2.5
    [microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2_5TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2_5VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 2048):
            The number of latent query tokens that represent the image features used in the text decoder component.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
    """

    model_type = "kosmos-2.5"
    sub_configs = {"text_config": Kosmos2_5TextConfig, "vision_config": Kosmos2_5VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    latent_query_num: int = 2048
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Kosmos2_5TextConfig()
            logger.info("`text_config` is `None`. initializing the `Kosmos2_5TextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = Kosmos2_5TextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Kosmos2_5VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Kosmos2_5VisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Kosmos2_5VisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["Kosmos2_5Config"]
