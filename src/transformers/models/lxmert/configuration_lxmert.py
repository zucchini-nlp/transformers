# Copyright 2018, Hao Tan, Mohit Bansal
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
"""LXMERT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class LxmertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LxmertModel`]. It is used
    to instantiate a LXMERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the Lxmert
    [unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the LXMERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LxmertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_qa_labels (`int`, *optional*, defaults to 9500):
            This represents the total number of different question answering (QA) labels there are. If using more than
            one dataset with QA, the user will need to account for the total number of labels that all of the datasets
            have in total.
        num_object_labels (`int`, *optional*, defaults to 1600):
            This represents the total number of semantically unique objects that lxmert will be able to classify a
            pooled-object feature as belonging too.
        num_attr_labels (`int`, *optional*, defaults to 400):
            This represents the total number of semantically unique attributes that lxmert will be able to classify a
            pooled-object feature as possessing.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`BertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        l_layers (`int`, *optional*, defaults to 9):
            Number of hidden layers in the Transformer language encoder.
        x_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer cross modality encoder.
        r_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer visual encoder.
        visual_feat_dim (`int`, *optional*, defaults to 2048):
            This represents the last dimension of the pooled-object features used as input for the model, representing
            the size of each object feature itself.
        visual_pos_dim (`int`, *optional*, defaults to 4):
            This represents the number of spatial features that are mixed into the visual features. The default is set
            to 4 because most commonly this will represent the location of a bounding box. i.e., (x, y, width, height)
        visual_loss_normalizer (`float`, *optional*, defaults to 6.67):
            This represents the scaling factor in which each visual loss is multiplied by if during pretraining, one
            decided to train with multiple vision-based loss objectives.
        task_matched (`bool`, *optional*, defaults to `True`):
            This task is used for sentence-image matching. If the sentence correctly describes the image the label will
            be 1. If the sentence does not correctly describe the image, the label will be 0.
        task_mask_lm (`bool`, *optional*, defaults to `True`):
            Whether or not to add masked language modeling (as used in pretraining models such as BERT) to the loss
            objective.
        task_obj_predict (`bool`, *optional*, defaults to `True`):
            Whether or not to add object prediction, attribute prediction and feature regression to the loss objective.
        task_qa (`bool`, *optional*, defaults to `True`):
            Whether or not to add the question-answering loss to the objective
        visual_obj_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to calculate the object-prediction loss objective
        visual_attr_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to calculate the attribute-prediction loss objective
        visual_feat_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to calculate the feature-regression loss objective
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
    """

    model_type = "lxmert"
    attribute_map = {}

    vocab_size: int = 30522
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_qa_labels: int = 9500
    num_object_labels: int = 1600
    num_attr_labels: int = 400
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    l_layers: int = 9
    x_layers: int = 5
    r_layers: int = 5
    visual_feat_dim: int = 2048
    visual_pos_dim: int = 4
    visual_loss_normalizer: float = 6.67
    task_matched: bool = True
    task_mask_lm: bool = True
    task_obj_predict: bool = True
    task_qa: bool = True
    visual_obj_loss: bool = True
    visual_attr_loss: bool = True
    visual_feat_loss: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_hidden_layers = {"vision": self.r_layers, "cross_encoder": self.x_layers, "language": self.l_layers}
        super().__post_init__(**kwargs)


__all__ = ["LxmertConfig"]
