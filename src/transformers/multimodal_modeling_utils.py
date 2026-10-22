# Copyright 2026 The HuggingFace Inc. team.
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

"""Collection of utils to be used by multimodal models."""

import enum
import math
from typing import Any

import torch

from .modeling_outputs import BaseModelOutputWithPooling
from .processing_utils import Unpack
from .utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from .utils.generic import accepts_precomputed_kwargs


logger = logging.get_logger(__name__)

MODALITIES = ("image", "video", "audio")

class MultimodalModelMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for modality in MODALITIES:
            getter = cls.__dict__.get(f"get_{modality}_features")
            if getter is None or getattr(getter, "_mm_wrapped", False):
                continue
            wrapped_getter = accepts_precomputed_kwargs(modality)(
                can_return_tuple(auto_docstring(getter))
            )
            wrapped_getter._mm_wrapped = True
            setattr(cls, f"get_{modality}_features", wrapped_getter)

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        raise NotImplementedError

    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        raise NotImplementedError

    def get_audio_features(
        self, input_features: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        raise NotImplementedError

    def _get_multimodal_mask(self, input_ids=None, inputs_embeds=None, token_ids=None) -> torch.BoolTensor:
        """[batch, seq] bool mask of placeholder positions. Default: every modality token the config defines."""
        if token_ids is None:
            token_ids = [
                token_id
                for modality in MODALITIES
                if (token_id := getattr(self.config, f"{modality}_token_id", None)) is not None
            ]

        # Loop over each modality for compile-friendliness
        if input_ids is not None:
            multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in token_ids:  # python ints: compiles to constant comparisons, no tensor construction
                multimodal_mask |= input_ids == token_id
            return multimodal_mask

        multimodal_mask = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device)
        for token_id in token_ids:
            token_embedding = self.get_input_embeddings()(torch.tensor(token_id, device=inputs_embeds.device))
            multimodal_mask |= (inputs_embeds == token_embedding).all(-1)
        return multimodal_mask

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        special_token_id: int,
        encoded_features: torch.FloatTensor | None = None,
    ) -> torch.BoolTensor:
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        placeholder_mask = self._get_multimodal_mask(input_ids, inputs_embeds, token_ids=[special_token_id])
        if encoded_features is not None:
            num_placeholder_tokens = placeholder_mask.sum()
            torch_compilable_check(
                num_placeholder_tokens * inputs_embeds.shape[-1] == encoded_features.numel(),
                f"Features and placeholder tokens do not match, tokens: {num_placeholder_tokens}, "
                f"features: {math.prod(encoded_features.shape[:-1])}",
            )
        return placeholder_mask.unsqueeze(-1).to(inputs_embeds.device)

    def merge_multimodal_embeddings(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        image_kwargs: dict[str, Any] | None = None,
        video_kwargs: dict[str, Any] | None = None,
        audio_kwargs: dict[str, Any] | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs,
    ):
        inputs = {
            "image": (pixel_values, image_kwargs or {}),
            "video": (pixel_values_videos, video_kwargs or {}),
            "audio": (input_features, audio_kwargs or {}),
        }
        mm_encoder_outputs = dict(mm_encoder_outputs or {})

        if any(
            mm_encoder_outputs.get(modality) is not None and raw_data is not None
            for modality, (raw_data, kwargs) in inputs.items()
        ):
            raise ValueError("Pass either raw multimodal inputs or precomputed `mm_encoder_outputs`, not both")

        # FIXME: must happen inside `get_audio_features` - not consistrnt atm across models
        # to udpate: gemma3n, gemma4, graniteSpeech-all
        # Strip padding tokens: only keep real (non-padding) audio soft tokens.
        # audio_mask_from_encoder = mm_encoder_outputs["audio"].attention_mask
        # audio_features = audio_features[audio_mask_from_encoder.to(audio_embeds.device)]

        for modality, (raw_inputs, mod_kwargs) in inputs.items():
            if mm_encoder_outputs.get(modality) is None and raw_inputs is not None:
                mm_encoder_outputs[modality] = getattr(self, f"get_{modality}_features")(
                    raw_inputs, **mod_kwargs, **kwargs, return_dict=True
                )
            if mm_encoder_outputs.get(modality) is None:
                continue

            embeds = mm_encoder_outputs[modality].pooler_output
            if isinstance(embeds, (list, tuple)):
                embeds = torch.cat(embeds, dim=0)
            embeds = embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            mask = self.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                special_token_id=getattr(self.config, f"{modality}_token_id"),
                encoded_features=embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(mask, embeds)
        return inputs_embeds, mm_encoder_outputs
