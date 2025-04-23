# Copyright 2022 The HuggingFace Inc. team.
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
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .image_processing_base import BatchFeature, ImageProcessingMixin
from .image_transforms import center_crop, normalize, rescale
from .image_utils import ChannelDimension, SizeDict, get_image_size, validate_preprocess_arguments
from .utils import is_torch_available, is_torchvision_available, is_vision_available, logging
from .utils.import_utils import requires


if is_torch_available():
    pass

if is_torchvision_available():
    from .image_utils import pil_torch_interpolation_mapping
else:
    pil_torch_interpolation_mapping = None

if is_vision_available():
    from .image_utils import PILImageResampling

logger = logging.get_logger(__name__)


INIT_SERVICE_KWARGS = [
    "processor_class",
    "image_processor_type",
]


@dataclass
class ImageProcessorConfig:
    def __init__(
        self,
        do_resize: Optional[bool],
        size: Optional[dict[str, int]],
        size_divisor: Optional[int],
        crop_size: Optional[dict[str, int]],
        resample: Optional[Union["PILImageResampling", int]],
        do_rescale: Optional[bool],
        rescale_factor: Optional[float],
        do_normalize: Optional[bool],
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_pad: Optional[bool],
        pad_size: Optional[dict[str, int]],
        do_center_crop: Optional[bool],
        default_to_square: Optional[bool],
        data_format: Optional[ChannelDimension],
        input_data_format: Optional[Union[str, ChannelDimension]],
        device: Optional[str],
    ):
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.do_center_crop = do_center_crop
        self.default_to_square = default_to_square
        self.data_format = data_format
        self.input_data_format = input_data_format
        self.device = device

        self.post_init()
        self.validate()

    def post_init(self):
        if self.size is not None:
            self.size = SizeDict(**get_size_dict(size=self.size, default_to_square=self.default_to_square))
        if self.crop_size is not None:
            self.crop_size = SizeDict(**get_size_dict(self.crop_size, param_name="crop_size"))

        if isinstance(self.image_mean, list):
            self.image_mean = tuple(self.image_mean)
        if isinstance(self.image_std, list):
            self.image_std = tuple(self.image_std)

        self.interpolation = (
            pil_torch_interpolation_mapping[self.resample]
            if isinstance(self.resample, (int, PILImageResampling))
            else self.resample
        )

        if self.do_rescale and self.do_normalize:
            # Fused rescale and normalize
            self.image_mean = np.array(self.image_mean) * (1.0 / self.rescale_factor)
            self.image_std = np.array(self.image_std) * (1.0 / self.rescale_factor)
            self.do_rescale = False

    def validate(self):
        validate_preprocess_arguments(
            do_rescale=self.do_rescale,
            rescale_factor=self.rescale_factor,
            do_normalize=self.do_normalize,
            image_mean=self.image_mean,
            image_std=self.image_std,
            do_pad=self.do_pad,
            size_divisibility=self.size_divisibility,
            do_center_crop=self.do_center_crop,
            crop_size=self.crop_size,
            do_resize=self.do_resize,
            size=self.size,
            resample=self.resample,
        )

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a BaseWatermarkingConfig instance from a dictionary of parameters.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            BaseWatermarkingConfig: Instance of BaseWatermarkingConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def __iter__(self):
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.

        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    def update(self, **kwargs):
        """
        Update the configuration attributes with new values.

        Args:
            **kwargs: Keyword arguments representing configuration attributes and their new values.
        """
        unused_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                unused_kwargs[key] = value
        self.validate()
        return unused_kwargs


@requires(backends=("vision",))
class BaseImageProcessor(ImageProcessingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, images, **kwargs) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs) -> BatchFeature:
        raise NotImplementedError("Each image processor must implement its own preprocess method")

    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The rescaled image.
        """
        return rescale(image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `Iterable[float]`):
                Image mean to use for normalization.
            std (`float` or `Iterable[float]`):
                Image standard deviation to use for normalization.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The normalized image.
        """
        return normalize(
            image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

    def center_crop(
        self,
        image: np.ndarray,
        size: dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        return center_crop(
            image,
            size=(size["height"], size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        return encoder_dict


VALID_SIZE_DICT_KEYS = (
    {"height", "width"},
    {"shortest_edge"},
    {"shortest_edge", "longest_edge"},
    {"longest_edge"},
    {"max_height", "max_width"},
)


def is_valid_size_dict(size_dict):
    if not isinstance(size_dict, dict):
        return False

    size_dict_keys = set(size_dict.keys())
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        if size_dict_keys == allowed_keys:
            return True
    return False


def convert_to_size_dict(
    size, max_size: Optional[int] = None, default_to_square: bool = True, height_width_order: bool = True
):
    # By default, if size is an int we assume it represents a tuple of (size, size).
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
        return {"height": size, "width": size}
    # In other configs, if size is an int and default_to_square is False, size represents the length of
    # the shortest edge after resizing.
    elif isinstance(size, int) and not default_to_square:
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict
    # Otherwise, if size is a tuple it's either (height, width) or (width, height)
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {"height": size[0], "width": size[1]}
    elif isinstance(size, (tuple, list)) and not height_width_order:
        return {"height": size[1], "width": size[0]}
    elif size is None and max_size is not None:
        if default_to_square:
            raise ValueError("Cannot specify both default_to_square=True and max_size")
        return {"longest_edge": max_size}

    raise ValueError(f"Could not convert size input to size dict: {size}")


def get_size_dict(
    size: Union[int, Iterable[int], dict[str, int]] = None,
    max_size: Optional[int] = None,
    height_width_order: bool = True,
    default_to_square: bool = True,
    param_name="size",
) -> dict:
    """
    Converts the old size parameter in the config into the new dict expected in the config. This is to ensure backwards
    compatibility with the old image processor configs and removes ambiguity over whether the tuple is in (height,
    width) or (width, height) format.

    - If `size` is tuple, it is converted to `{"height": size[0], "width": size[1]}` or `{"height": size[1], "width":
    size[0]}` if `height_width_order` is `False`.
    - If `size` is an int, and `default_to_square` is `True`, it is converted to `{"height": size, "width": size}`.
    - If `size` is an int and `default_to_square` is False, it is converted to `{"shortest_edge": size}`. If `max_size`
      is set, it is added to the dict as `{"longest_edge": max_size}`.

    Args:
        size (`Union[int, Iterable[int], Dict[str, int]]`, *optional*):
            The `size` parameter to be cast into a size dictionary.
        max_size (`Optional[int]`, *optional*):
            The `max_size` parameter to be cast into a size dictionary.
        height_width_order (`bool`, *optional*, defaults to `True`):
            If `size` is a tuple, whether it's in (height, width) or (width, height) order.
        default_to_square (`bool`, *optional*, defaults to `True`):
            If `size` is an int, whether to default to a square image or not.
    """
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
        logger.info(
            f"{param_name} should be a dictionary on of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size}."
            f" Converted to {size_dict}.",
        )
    else:
        size_dict = size

    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}"
        )
    return size_dict


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_patch_output_size(image, target_resolution, input_data_format):
    """
    Given an image and a target resolution, calculate the output size of the image after cropping to the target
    """
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width
