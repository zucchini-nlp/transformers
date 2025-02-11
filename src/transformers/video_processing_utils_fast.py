# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict, Union

import numpy as np

from .image_processing_utils import (
    BatchFeature,
    get_size_dict,
)
from .image_transforms import (
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
)
from .image_utils import (
    ChannelDimension,
    SizeDict,
    get_image_size_for_max_height_width,
    validate_fast_preprocess_arguments,
    validate_kwargs,
)
from .utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from .video_processing_utils import BaseVideoProcessor
from .video_utils import (
    VideoInput,
    group_videos_by_shape,
    make_batched_videos,
    reorder_videos,
    to_channel_dimension_format,
)


if is_vision_available():
    from .image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


@dataclass
class FastVideoProcessorConfig:
    do_convert_rgb: bool = None,
    size: Optional[Tuple[int, int]] = None
    crop_size: Optional[Tuple[int, int]] = None
    do_resize: bool = True
    do_center_crop: bool = False
    do_rescale: bool = True
    do_normalize: bool = True
    rescale_factor: float = 1 / 255
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    resample: str = "bilinear"
    return_tensors: Optional[str] = None
    input_data_format: Optional[ChannelDimension] = (None,)
    data_format: Optional[ChannelDimension] = (ChannelDimension.FIRST,)
    device: Optional["torch.device"] = (None,)
    _extra_kwargs = {}  # Track dynamically added args

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra_kwargs[key] = value

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra_kwargs[key] = value

    def validate(self):
        pass

    def get_all_kwargs(self):
        return {**self.__dict__, **self._extra_kwargs}

    def finalize(self):
        """Ensure proper data types and prepare for processing."""
        if isinstance(self.image_mean, list):
            self.image_mean = tuple(self.image_mean)
        if isinstance(self.image_std, list):
            self.image_std = tuple(self.image_std)

        self.size = (
            get_size_dict(size=self.size, default_to_square=self.default_to_square) if self.size is not None else None
        )
        self.crop_size = get_size_dict(self.crop_size, param_name="crop_size") if self.crop_size is not None else None

        if self.do_rescale and self.do_normalize:
            self.image_mean = torch.tensor(self.image_mean, device=self.device) * (1.0 / self.rescale_factor)
            self.image_std = torch.tensor(self.image_std, device=self.device) * (1.0 / self.rescale_factor)


class DefaultFastVideoProcessorInitKwargs(TypedDict, total=False):
    do_resize: Optional[bool]
    size: Optional[Dict[str, int]]
    default_to_square: Optional[bool]
    resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]]
    do_center_crop: Optional[bool]
    crop_size: Optional[Dict[str, int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[Union[int, float]]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, List[float]]]
    image_std: Optional[Union[float, List[float]]]
    do_convert_rgb: Optional[bool]


BASE_VIDEO_PROCESSOR_FAST_DOCSTRING = r"""
    Args:
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the video's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `self.size`):
            Size of the output videoafter resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square videowhen resizing, if size is an int.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the videoto the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to `self.crop_size`):
            Size of the output videoafter applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the videoby the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
            Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Mean to use if normalizing the video. This is a float or list of floats the length of the number of
            channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
            number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `self.image_std`):
            Whether to convert the videoto RGB."""

BASE_VIDEO_PROCESSOR_FAST_DOCSTRING_PREPROCESS = r"""
    Preprocess a video or batch of videos.

    Args:
        videos (`VideoInput`):
            Image to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
            passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the video.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Describes the maximum input dimensions to the model.
        resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the video. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the video.
        crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
            Size of the output videoafter applying `center_crop`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the video.
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the videoby if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the video.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the videoto RGB.
        return_tensors (`str` or `TensorType`, *optional*):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format for the output video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: videoin (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: videoin (height, width, num_channels) format.
            - Unset: Use the channel dimension format of the input video.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input video. If unset, the channel dimension format is inferred
            from the input video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: videoin (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: videoin (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: videoin (height, width) format.
        device (`torch.device`, *optional*):
            The device to process the videos on. If unset, the device is inferred from the input videos."""


@add_start_docstrings(
    "Constructs a fast base VideoProcessor.",
    BASE_VIDEO_PROCESSOR_FAST_DOCSTRING,
)
class BaseVideoProcessorFast(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos"]
    extra_kwargs_for_processing = {}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = FastVideoProcessorConfig(**kwargs, _extra_kwargs=self.extra_kwargs_for_processing)
        for key in vars(self.config).keys():
            setattr(self, key, getattr(self.config, key))

    def resize(
        self,
        video: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize a video to `(size["height"], size["width"])`.

        Args:
            video (`torch.Tensor`):
                Video to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output video.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the video e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized video.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the video so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original video.
            new_size = get_size_with_aspect_ratio(
                video.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                video,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(video.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        return F.resize(video, new_size, interpolation=interpolation)

    def rescale(
        self,
        video: "torch.Tensor",
        scale: float,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Rescale a video by a scale factor. video = video * scale.

        Args:
            video (`torch.Tensor`):
                Video to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.

        Returns:
            `torch.Tensor`: The rescaled video.
        """
        return video * scale

    def normalize(
        self,
        video: "torch.Tensor",
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        **kwargs,
    ) -> "torch.Tensor":
        """
        Normalize a video. video = (video - mean) / std.

        Args:
            video (`torch.Tensor`):
                video to normalize.
            mean (`torch.Tensor`, `float` or `Iterable[float]`):
                video mean to use for normalization.
            std (`torch.Tensor`, `float` or `Iterable[float]`):
                video standard deviation to use for normalization.

        Returns:
            `torch.Tensor`: The normalized video.
        """
        return F.normalize(video, mean, std)

    def rescale_and_normalize(
        self,
        videos: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
    ) -> "torch.Tensor":
        """
        Rescale and normalize videos.
        """
        if do_rescale and do_normalize:
            videos = self.normalize(videos.to(dtype=torch.float32), image_mean, image_std)
        elif do_rescale:
            videos = videos * rescale_factor
        elif do_normalize:
            videos = self.normalize(videos, image_mean, image_std)

        return videos

    def center_crop(
        self,
        video: "torch.Tensor",
        size: Dict[str, int],
        **kwargs,
    ) -> "torch.Tensor":
        """
        Center crop a video to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the video is padded with 0's and then center cropped.

        Args:
            video (`"torch.Tensor"`):
                Video to center crop.
            size (`Dict[str, int]`):
                Size of the output video.

        Returns:
            `torch.Tensor`: The center cropped video.
        """
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        return F.center_crop(video, (size["height"], size["width"]))

    def convert_to_rgb(
        self,
        video: "torch.Tensor",
    ) -> VideoInput:
        """
        Converts a video to RGB format.

        Args:
            video (`"torch.Tensor"`):
                The video to convert.

        Returns:
            `torch.Tensor`: The converted video.
        """

        video = F.grayscale_to_rgb(video)
        if video.shape[-3] == 3 or not (video[..., 3, :, :] < 255).any():
            return video

        # There is a transparency layer, blend it with a white background.
        # Calculate the alpha proportion for blending.
        alpha = video[..., 3, :, :] / 255.0
        video = (1 - alpha[..., None, :, :]) * 255 + alpha[..., None, :, :] * video[..., :3, :, :]
        return video

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input videos for processing.
        """
        videos = make_batched_videos(videos)
        processed_videos = []
        for video in videos:
            if isinstance(video, np.ndarray):
                video = to_channel_dimension_format(video, ChannelDimension.FIRST, input_data_format)
                # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
                video = torch.from_numpy(video).contiguous()

            # Now that we have torch tensors, we can move them to the right device
            if device is not None:
                video = video.to(device)
            processed_videos.append(video)

        return processed_videos

    def _prepare_process_arguments(self, **kwargs) -> tuple:
        """
        Prepare the arguments for the process method.
        """
        config = copy.deepcopy(self.config)
        validate_kwargs(
            valid_processor_keys=config.get_all_kwargs().keys(),
            captured_kwargs=kwargs.keys(),
        )

        # Update config with new values
        config.update(**kwargs)
        config.finalize()

        validate_fast_preprocess_arguments(
            do_rescale=config.do_rescale,
            rescale_factor=config.rescale_factor,
            do_normalize=config.do_normalize,
            image_mean=config.image_mean,
            image_std=config.image_std,
            do_resize=config.do_resize,
            size=config.size,
            do_center_crop=config.do_center_crop,
            crop_size=config.crop_size,
            resample=config.resample,
            return_tensors=config.return_tensors,
            data_format=config.data_format,
        )

        return config

    def preprocess(
        self,
        videos: List["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        resample: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[ChannelDimension] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        device: Optional["torch.device"] = None,
        **kwargs,
    ) -> BatchFeature:
        # Prepare kwargs for preprocessing: convert to tensor, validate and move to device if needed
        config = self._prepare_process_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
            device=device,
            **kwargs,
        )

        # Prepare videos by conveting to correctly batched list and move to device
        videos = self._prepare_input_videos(
            videos=videos, input_data_format=config.input_data_format, device=config.device
        )

        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if config.do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if config.do_resize:
                stacked_videos = self.resize(
                    video=stacked_videos, size=config.size, interpolation=config.interpolation
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if config.do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, config.crop_size)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                config.do_rescale,
                config.rescale_factor,
                config.do_normalize,
                config.image_mean,
                config.image_std,
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_videos = torch.stack(processed_videos, dim=0) if config.return_tensors else processed_videos

        return BatchFeature(data={"pixel_values_videos": processed_videos}, tensor_type=config.return_tensors)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        return encoder_dict
