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
"""AutoAudioProcessor class."""

import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from ...audio_processing_utils import AudioProcessorBase

# Build the list of all audio processors
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import (
    AUDIO_PROCESSOR_NAME,
    CONFIG_NAME,
    cached_file,
    logging,
)
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    AUDIO_PROCESSOR_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    AUDIO_PROCESSOR_MAPPING_NAMES = OrderedDict(
        [
            ("audio-spectrogram-transformer", "ASTAudioProcessor"),
            ("clap", "ClapAudioProcessor"),
            ("dac", "DacAudioProcessor"),
            ("data2vec-audio", "Wav2Vec2AudioProcessor"),
            ("encodec", "EncodecAudioProcessor"),
            ("granite_speech", "GraniteSpeechAudioProcessor"),
            ("hubert", "Wav2Vec2AudioProcessor"),
            ("mimi", "EncodecAudioProcessor"),
            ("moonshine", "Wav2Vec2AudioProcessor"),
            ("moshi", "EncodecAudioProcessor"),
            ("perceiver", "PerceiverAudioProcessor"),
            ("phi4_multimodal", "Phi4MultimodalAudioProcessor"),
            ("pop2piano", "Pop2PianoAudioProcessor"),
            ("seamless_m4t", "SeamlessM4TAudioProcessor"),
            ("seamless_m4t_v2", "SeamlessM4TAudioProcessor"),
            ("sew", "Wav2Vec2AudioProcessor"),
            ("sew-d", "Wav2Vec2AudioProcessor"),
            ("speech_to_text", "Speech2TextAudioProcessor"),
            ("speecht5", "SpeechT5AudioProcessor"),
            ("swin", "ViTAudioProcessor"),
            ("swinv2", "ViTAudioProcessor"),
            ("unispeech", "Wav2Vec2AudioProcessor"),
            ("unispeech-sat", "Wav2Vec2AudioProcessor"),
            ("univnet", "UnivNetAudioProcessor"),
            ("wav2vec2", "Wav2Vec2AudioProcessor"),
            ("wav2vec2-bert", "Wav2Vec2AudioProcessor"),
            ("wav2vec2-conformer", "Wav2Vec2AudioProcessor"),
            ("wavlm", "Wav2Vec2AudioProcessor"),
            ("whisper", "WhisperAudioProcessor"),
        ]
    )

AUDIO_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, AUDIO_PROCESSOR_MAPPING_NAMES)


def audio_processor_class_from_name(class_name: str):
    for module_name, extractors in AUDIO_PROCESSOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for _, extractor in AUDIO_PROCESSOR_MAPPING._extra_content.items():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_audio_processor_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the audio processor configuration from a pretrained model audio processor configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the audio processor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the audio processor.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    audio_processor_config = get_audio_processor_config("facebook/wav2vec2-base-960h")
    # This model does not have a audio processor config so the result will be an empty dict.
    audio_processor_config = get_audio_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained audio processor locally and you can reload its config
    from transformers import AutoAudioProcessor

    audio_processor = AutoAudioProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_processor.save_pretrained("audio-processor-test")
    audio_processor = get_audio_processor_config("audio-processor-test")
    ```"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        AUDIO_PROCESSOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved_config_file is None:
        logger.info(
            "Could not locate the audio processor configuration file, will try to use the model config instead."
        )
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


class AutoAudioProcessor:
    r"""
    This is a generic audio processor class that will be instantiated as one of the audio processor classes of the
    library when created with the [`AutoAudioProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoAudioProcessor is designed to be instantiated "
            "using the `AutoAudioProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(AUDIO_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the audio processor classes of the library from a pretrained model vocabulary.

        The audio processor class to instantiate is selected based on the `model_type` property of the config object
        (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained audio_processor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a audio processor file saved using the
                  [`~audio_processing_utils.AudioProcessorBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved audio processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model audio processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the audio processor files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final audio processor object. If `True`, then this
                functions returns a `Tuple(audio_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not audio processor attributes: i.e., the part of
                `kwargs` which has not been used to update `audio_processor` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are audio processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* audio_processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import AutoAudioProcessor

        >>> # Download audio processor from huggingface.co and cache.
        >>> audio_processor = AutoAudioProcessor.from_pretrained("facebook/wav2vec2-base-960h")

        >>> # If audio processor files are in a directory (e.g. audio processor was saved using *save_pretrained('./test/saved_model/')*)
        >>> # audio_processor = AutoAudioProcessor.from_pretrained("./test/saved_model/")
        ```"""
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        config_dict, _ = AudioProcessorBase.get_audio_processor_dict(pretrained_model_name_or_path, **kwargs)
        audio_processor_class = config_dict.get("audio_processor_type", None)
        audio_processor_auto_map = None
        if "AutoAudioProcessor" in config_dict.get("auto_map", {}):
            audio_processor_auto_map = config_dict["auto_map"]["AutoAudioProcessor"]

        # If we still don't have the audio processor class, check if we're loading from a previous feature extractor config
        # and if so, infer the audio processor class from there.
        if audio_processor_class is None and audio_processor_auto_map is None:
            feature_extractor_class = config_dict.pop("feature_extractor_type", None)
            if feature_extractor_class is not None:
                audio_processor_class = feature_extractor_class.replace("FeatureExtractor", "AudioProcessor")
            if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
                feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]
                audio_processor_auto_map = feature_extractor_auto_map.replace("FeatureExtractor", "AudioProcessor")

        # If we don't find the audio processor class in the audio processor config, let's try the model config.
        if audio_processor_class is None and audio_processor_auto_map is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            # It could be in `config.audio_processor_type``
            audio_processor_class = getattr(config, "audio_processor_type", None)
            if hasattr(config, "auto_map") and "AutoAudioProcessor" in config.auto_map:
                audio_processor_auto_map = config.auto_map["AutoAudioProcessor"]

        if audio_processor_class is not None:
            audio_processor_class = audio_processor_class_from_name(audio_processor_class)

        has_remote_code = audio_processor_auto_map is not None
        has_local_code = audio_processor_class is not None or type(config) in AUDIO_PROCESSOR_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = audio_processor_auto_map
            audio_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                audio_processor_class.register_for_auto_class()
            return audio_processor_class.from_dict(config_dict, **kwargs)
        elif audio_processor_class is not None:
            return audio_processor_class.from_dict(config_dict, **kwargs)
        # Last try: we use the AUDIO_PROCESSOR_MAPPING.
        elif type(config) in AUDIO_PROCESSOR_MAPPING:
            audio_processor_class = AUDIO_PROCESSOR_MAPPING[type(config)]
            return audio_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        raise ValueError(
            f"Unrecognized audio processor in {pretrained_model_name_or_path}. Should have a "
            f"`audio_processor_type` key in its {AUDIO_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in AUDIO_PROCESSOR_MAPPING_NAMES.keys())}"
        )

    @staticmethod
    def register(
        config_class,
        audio_processor_class,
        exist_ok=False,
    ):
        """
        Register a new audio processor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            audio_processor_class ([`AudioProcessorBase`]):
                The audio processor to register.
        """
        AUDIO_PROCESSOR_MAPPING.register(config_class, audio_processor_class, exist_ok=exist_ok)


__all__ = ["AUDIO_PROCESSOR_MAPPING", "AutoAudioProcessor"]
