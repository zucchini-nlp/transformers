# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Fast Image processor class for Qwen2-VL."""

import copy

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    VideoInput,
    get_image_size,
    make_batched_videos,
    make_flat_list_of_images,
    valid_images,
)
from ...utils import (
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)
from .image_processing_qwen2_vl import Qwen2VLImageProcessorConfig, smart_resize


if is_torch_available():
    import torch


if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


@add_start_docstrings(
    "Constructs a fast Qwen2-VL image processor that dynamically resizes images based on the original images.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """,
)
class Qwen2VLImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs):
        config = Qwen2VLImageProcessorConfig(**kwargs)
        kwargs = config.filter_out_unused_kwargs(kwargs)

        if config.size is not None and ("shortest_edge" not in config.size or "longest_edge" not in config.size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if config.min_pixels is not None:
            config.size["shortest_edge"] = config.min_pixels
        if config.max_pixels is not None:
            config.size["longest_edge"] = config.max_pixels
        super().__init__(config, **kwargs)

    def _preprocess(self, images: ImageInput, config: Qwen2VLImageProcessorConfig) -> BatchFeature:
        # Prepare images
        images = self._prepare_input_images(
            images=images,
            do_convert_rgb=config.do_convert_rgb,
            input_data_format=config.input_data_format,
            device=config.device,
        )

        height, width = get_image_size(images[0], channel_dim=ChannelDimension.FIRST)
        resized_height, resized_width = height, width

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if config.do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=config.patch_size * config.merge_size,
                    min_pixels=config.size["shortest_edge"],
                    max_pixels=config.size["longest_edge"],
                )
                stacked_images = F.resize(
                    stacked_images, size=(resized_height, resized_width), interpolation=config.interpolation
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images,
                config.do_rescale,
                config.rescale_factor,
                config.do_normalize,
                config.fused_image_mean,
                config.fused_image_std,
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        patches = torch.stack(processed_images, dim=0)

        # Flatten patches to (seq-len, patch-dim)
        temporal_patch_size = config.temporal_patch_size
        patch_size = config.patch_size
        merge_size = config.merge_size
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = patches[-1].unsqueeze(0).repeat(temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = patches.view(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        return_tensors=None,
        **kwargs,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            min_pixels (`int`, *optional*, defaults to `self.min_pixels`):
                The min pixels of the image to resize the image.
            max_pixels (`int`, *optional*, defaults to `self.max_pixels`):
                The max pixels of the image to resize the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spacial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            device (`torch.device`, *optional*):
                The device to process the images on. If unset, the device is inferred from the input images.
        """
        config = copy.deepcopy(self.config)
        unused_kwargs = config.update(**kwargs)
        logger.warning(f"Some kwargs are not used in `__call__`: {unused_kwargs.keys()}")

        # Extra checks for only Fast Image Processor
        if return_tensors is not None and return_tensors != "pt":
            raise ValueError("Only returning PyTorch tensors is currently supported.")

        if config.data_format is not None and config.data_format != ChannelDimension.FIRST:
            raise ValueError("Only channel first data format is currently supported.")

        if config.min_pixels is not None and config.max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            config.size = {"shortest_edge": config.min_pixels, "longest_edge": config.max_pixels}

        if images is not None:
            images = make_flat_list_of_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(image, config)
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = torch.stack(pixel_values)
            vision_grid_thws = torch.tensor(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(images, config)
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = torch.stack(pixel_values)
            vision_grid_thws = torch.tensor(vision_grid_thws)
            data = {"pixel_values_videos": pixel_values, "video_grid_thw": vision_grid_thws}

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Qwen2VLImageProcessorFast"]
