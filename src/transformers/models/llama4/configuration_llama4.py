# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Llama4VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4VisionModel`]. It is used to instantiate a
    Llama4 vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        num_hidden_layers (`int`, *optional*, defaults to 34):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        vision_output_dim (`int`, *optional*, defaults to 7680):
            Dimensionality of the vision model output. Includes output of transformer
            encoder with intermediate layers and global transformer encoder.
        image_size (`int`, *optional*, defaults to 448):
            The size (resolution) of each image *tile*.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            Controls which vision tokens are kept from the backbone. `"default"` drops the CLS token and `"full"` keeps all tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pixel_shuffle_ratio (`float`, *optional*, defaults to 0.5):
            Pixel-shuffle ratio for downsampling patch tokens. Smaller values produce fewer tokens (more downsampling).
        projector_input_dim (`int`, *optional*, defaults to 4096):
            Width of the vision adapter MLP before pixel shuffle. Larger value increases capacity and compute.
        projector_output_dim (`int`, *optional*, defaults to 4096):
            Output width of the vision adapter. Larger value yields higher-dimensional image features.
        multi_modal_projector_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the multi-modal projector layers.
        projector_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate inside the vision adapter MLP. Higher value adds more regularization.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate on vision attention probabilities. Higher value adds more regularization.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE Parameters
    """

    base_model_tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "vision_adapter.mlp.fc1": "colwise",
        "vision_adapter.mlp.fc2": "rowwise",
        "patch_embedding.linear": "colwise_gather_output",
    }
    model_type = "llama4_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    hidden_act: str = "gelu"
    num_hidden_layers: int = 34
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5632
    vision_output_dim: int = 7680
    image_size: int | list[int] | tuple[int, int] = 448
    patch_size: int | list[int] | tuple[int, int] = 14
    norm_eps: float = 1e-5
    vision_feature_select_strategy: str = "default"
    initializer_range: float = 0.02
    pixel_shuffle_ratio: float = 0.5
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096
    multi_modal_projector_bias: bool = False
    projector_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Llama4TextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4TextModel`]. It is used to instantiate a
    Llama4 text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 202048):
            Vocabulary size of the Llama4 text model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Llama4TextModel`].
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the embeddings and hidden states.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        intermediate_size_mlp (`int`, *optional*, defaults to 16384):
            Intermediate size of dense MLP layers. Larger value increases FFN capacity and compute.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If not
            specified, will default to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            Per-head attention dimension. Larger value increases head width and compute.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*, defaults to 128004):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning of sentence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end of sentence token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate on vision attention probabilities. Higher value adds more regularization.
        num_experts_per_tok (`int`, *optional*, defaults to 1):
            Top-k experts routed per token. Higher value uses more experts per token and more compute.
        num_local_experts (`int`, *optional*, defaults to 16):
            Number of experts in each MoE layer. Higher value increases capacity and routing choices.
        moe_layers (`list[int]`, *optional*):
            List of layer indices that use MoE. Overrides `interleave_moe_layer_step` when set.
        interleave_moe_layer_step (`int`, *optional*, defaults to 1):
            Spacing between MoE layers when `moe_layers` is `None`. Larger value means fewer MoE layers.
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to L2-normalize queries/keys on RoPE layers. Can stabilize attention when enabled.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to return router logits (and auxiliary loss) in outputs.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Weight for the router auxiliary loss. Higher value makes routing loss contribute more to total loss.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            Amount of noise added to router logits during training. Higher value increases exploration.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        no_rope_layers (`list[int]`, *optional*):
            List with at least the same length as the number of layers in the model.
            A `1` at an index position indicates that the corresponding layer will use RoPE,
            while a `0` indicates that it's a NoPE layer.
        no_rope_layer_interval (`int`, *optional*, defaults to 4):
            If `no_rope_layers` is `None`, it will be created using a NoPE layer every
            `no_rope_layer_interval` layers.
        attention_chunk_size (`int`, *optional*, defaults to 8192):
            Chunk size for the attention computation. Smaller value enforces more local attention and lowers memory.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attn_temperature_tuning (`bool`, *optional*, defaults to `True`):
            Whether to dynamically scale the attention temperature for each query token based on sequence length.
            Recommended for long sequences (e.g., >32k tokens) to maintain stable output results.
        floor_scale (`int`, *optional*, defaults to 8192):
            Base scale (in tokens) for attention temperature tuning. Larger value delays scaling to longer positions.
        attn_scale (`float`, *optional*, defaults to 0.1):
            Strength of attention temperature tuning. Larger value increases scaling at long positions.

    Example:
    """

    model_type = "llama4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.shared_expert.gate_proj": "colwise",
        "layers.*.feed_forward.shared_expert.up_proj": "colwise",
        "layers.*.feed_forward.shared_expert.down_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "packed_rowwise",  # row because not linear
        "layers.*.feed_forward.experts.down_proj": "colwise",  # col because not linear
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "grouped_gemm",  # row because not linear
        "layers.*.feed_forward.experts.down_proj": "grouped_gemm",  # col because not linear
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
        "layers.*.feed_forward.router": "ep_router",
    }

    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    moe_layers: list[int] | None = None
    interleave_moe_layer_step: int = 1
    use_qk_norm: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    rope_parameters: RopeParameters | dict | None = None
    no_rope_layers: list[int] | None = None
    no_rope_layer_interval: int = 4
    attention_chunk_size: int = 8192
    layer_types: list[int] | None = None
    attn_temperature_tuning: bool = True
    floor_scale: int = 8192
    attn_scale: float = 0.1
    attention_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        default_no_rope_layers = [
            int((layer_idx + 1) % self.no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
        ]
        self.no_rope_layers = self.no_rope_layers if self.no_rope_layers else default_no_rope_layers
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

        self.moe_layers = (
            self.moe_layers
            if self.moe_layers is not None
            else list(
                range(
                    self.interleave_moe_layer_step - 1,
                    self.num_hidden_layers,
                    self.interleave_moe_layer_step,
                )
            )
        )

        if self.layer_types is None:
            self.layer_types = [
                "chunked_attention" if no_rope else "full_attention" for no_rope in self.no_rope_layers
            ]

        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class Llama4Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4Model`]. It is used to instantiate an
    Llama4 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vision_config (`Llama4VisionConfig`, *optional*):
            The Llama4 Vision config.
        text_config (`Llama4TextConfig`, *optional*):
            The Llama4 Text config.
        boi_token_index (`int`, *optional*, defaults to 200080):
            The begin-of-image token index to wrap the image prompt.
        eoi_token_index (`int`, *optional*, defaults to 200081):
            The end-of-image token index to wrap the image prompt.
        image_token_index (`int`, *optional*, defaults to 200092):
            The image token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    ```python
    >>> from transformers import Llama4Model, Llama4Config

    >>> # Initializing a Llama4 7B style configuration
    >>> configuration = Llama4Config()

    >>> # Initializing a model from the Llama4 7B style configuration
    >>> model = Llama4Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama4"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
    sub_configs = {"text_config": Llama4TextConfig, "vision_config": Llama4VisionConfig}
    base_model_tp_plan = {
        "multi_modal_projector.linear_1": "colwise_rep",
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    boi_token_index: int = 200080
    eoi_token_index: int = 200081
    image_token_index: int = 200092
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = Llama4VisionConfig()
            logger.info("vision_config is None, using default llama4 vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Llama4VisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = Llama4TextConfig()
            logger.info("text_config is None, using default llama4 text config")
        elif isinstance(self.text_config, dict):
            self.text_config = Llama4TextConfig(**self.text_config)
        super().__post_init__(**kwargs)


__all__ = ["Llama4Config", "Llama4TextConfig", "Llama4VisionConfig"]
