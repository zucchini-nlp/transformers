# Copyright 2024 The HuggingFace Inc. team.
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
"""ColPali model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ColPaliConfig(PreTrainedConfig):
    r"""
    Configuration class to store the configuration of a [`ColPaliForRetrieval`]. It is used to instantiate an instance
    of `ColPaliForRetrieval` according to the specified arguments, defining the model architecture following the methodology
    from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Creating a configuration with the default settings will result in a configuration where the VLM backbone is set to the
    default PaliGemma configuration, i.e the one from [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2).

    Note that contrarily to what the class name suggests (actually the name refers to the ColPali **methodology**), you can
    use a different VLM backbone model than PaliGemma by passing the corresponding VLM configuration to the class constructor.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vlm_config (`PreTrainedConfig`, *optional*):
            Configuration of the VLM backbone model.
        text_config (`PreTrainedConfig`, *optional*):
            Configuration of the text backbone model. Overrides the `text_config` attribute of the `vlm_config` if provided.
        embedding_dim (`int`, *optional*, defaults to 128):
            Dimension of the multi-vector embeddings produced by the model.

    Example:

    ```python
    from transformers.models.colpali import ColPaliConfig, ColPaliForRetrieval

    config = ColPaliConfig()
    model = ColPaliForRetrieval(config)
    ```
    """

    model_type = "colpali"
    sub_configs = {"vlm_config": PreTrainedConfig, "text_config": AutoConfig}

    vlm_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    embedding_dim: int = 128

    def __post_init__(self, **kwargs):
        if self.vlm_config is None:
            self.vlm_config = CONFIG_MAPPING["paligemma"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `PaliGemmaConfig` with default values."
            )
        elif isinstance(self.vlm_config, dict):
            self.vlm_config = CONFIG_MAPPING[self.vlm_config["model_type"]](**self.vlm_config)

        self.text_config = self.text_config if self.text_config is not None else self.vlm_config.text_config
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "gemma")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)

        super().__post_init__(**kwargs)


__all__ = ["ColPaliConfig"]
