# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""BEiT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BeitConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BeitModel`]. It is used to instantiate an BEiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BEiT
    [microsoft/beit-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k) architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 8192):
            Vocabulary size of the BEiT model. Defines the number of different image tokens that can be used during
            pre-training.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to use BERT-style absolute position embeddings.
        use_relative_position_bias (`bool`, *optional*, defaults to `False`):
            Whether to use T5-style relative position embeddings in the self-attention layers.
        use_shared_relative_position_bias (`bool`, *optional*, defaults to `False`):
            Whether to use the same relative position embeddings across all self-attention layers of the Transformer.
        layer_scale_init_value (`float`, *optional*, defaults to 0.1):
            Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
            CLS token, before applying the classification head.
        pool_scales (`tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`):
            Pooling scales used in Pooling Pyramid Module applied on the last feature map.
        use_auxiliary_head (`bool`, *optional*, defaults to `True`):
            Whether to use an auxiliary head during training.
        auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
            Weight of the cross-entropy loss of the auxiliary head.
        auxiliary_channels (`int`, *optional*, defaults to 256):
            Number of channels to use in the auxiliary head.
        auxiliary_num_convs (`int`, *optional*, defaults to 1):
            Number of convolutional layers to use in the auxiliary head.
        auxiliary_concat_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the output of the auxiliary head with the input before the classification layer.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.
        out_features (`list[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        add_fpn (`bool`, *optional*, defaults to `False`):
            Whether to add a FPN as part of the backbone. Only relevant for [`BeitBackbone`].
        reshape_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
            case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
            seq_len, hidden_size)`. Only relevant for [`BeitBackbone`].

    Example:

    ```python
    >>> from transformers import BeitConfig, BeitModel

    >>> # Initializing a BEiT beit-base-patch16-224-pt22k style configuration
    >>> configuration = BeitConfig()

    >>> # Initializing a model (with random weights) from the beit-base-patch16-224-pt22k style configuration
    >>> model = BeitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "beit"

    vocab_size: int = 8192
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    use_mask_token: bool = False
    use_absolute_position_embeddings: bool = False
    use_relative_position_bias: bool = False
    use_shared_relative_position_bias: bool = False
    layer_scale_init_value: float = 0.1
    drop_path_rate: float = 0.1
    use_mean_pooling: bool = True
    pool_scales: list[int] | tuple[int, ...] = (1, 2, 3, 6)
    use_auxiliary_head: bool = True
    auxiliary_loss_weight: float = 0.4
    auxiliary_channels: int = 256
    auxiliary_num_convs: int = 1
    auxiliary_concat_input: bool = False
    semantic_loss_ignore_index: int = 255
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    add_fpn: bool = False
    reshape_hidden_states: bool = True

    def __post_init__(self, **kwargs):
        if "segmentation_indices" in kwargs and self._out_indices is None:
            self._out_indices = kwargs.pop("segmentation_indices")

        # backbone attributes
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )

        super().__post_init__(**kwargs)


__all__ = ["BeitConfig"]
