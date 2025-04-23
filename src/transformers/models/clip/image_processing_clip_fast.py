# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for CLIP."""

from ...image_processing_utils import ImageProcessorConfig
from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...utils import add_start_docstrings


class CLIPImageProcessorConfig(ImageProcessorConfig):
    def __init__(
        self,
        resample=PILImageResampling.BICUBIC,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        size={"shortest_edge": 224},
        default_to_square=False,
        crop_size={"height": 224, "width": 224},
        do_resize=True,
        do_center_crop=True,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        **kwargs,
    ):
        super().__init__(
            self,
            resample,
            image_mean,
            image_std,
            size,
            default_to_square,
            crop_size,
            do_resize,
            do_center_crop,
            do_rescale,
            do_normalize,
            do_convert_rgb,
            **kwargs,
        )


@add_start_docstrings(
    "Constructs a fast CLIP image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class CLIPImageProcessorFast(BaseImageProcessorFast):
    def __init__(self, **kwargs):
        self.config = CLIPImageProcessorConfig(**kwargs)


__all__ = ["CLIPImageProcessorFast"]
