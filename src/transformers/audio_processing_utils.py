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
"""
Audio processor saving/loading class for common audio processors.
"""

import copy
import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import FeatureExtractionMixin
from .utils import (
    AUDIO_PROCESSOR_NAME,
    FEATURE_EXTRACTOR_NAME,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_file,
    copy_func,
    download_url,
    is_offline_mode,
    is_remote_url,
    is_torch_available,
    logging,
)


if TYPE_CHECKING:
    if is_torch_available():
        import torch  # noqa


logger = logging.get_logger(__name__)


# Inherit from `FeatureExtractionMixin` for now and split out after long deprecation cycle
class AudioProcessorBase(FeatureExtractionMixin):
    """
    This is a base class for audio processors used to provide saving/loading functionality for audio processors.
    """

    _auto_class = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # In contrast to `FeatureExtractionMixin`, try to load config from AUDIO_PROCESSOR_NAME. If not exists, try FEATURE_EXTRACTOR_NAME
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        r"""
        Instantiate a type of [`~audio_processing_utils.AudioProcessorBase`] from an audio processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained audio processor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing an audio processor file saved using the
                  [`~audio_processing_utils.AudioProcessorBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved audio processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model audio processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the audio processor files and override the cached versions
                if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final audio processor object. If `True`, then this
                functions returns a `Tuple(audio_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not audio processor attributes: i.e., the part of
                `kwargs` which has not been used to update `audio_processor` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are audio processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* audio processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            An audio processor of type [`~audio_processing_utils.AudioProcessorBase`].

        Examples:

        ```python
        # We can't instantiate directly the base class *AudioProcessorBase* so let's show the examples on a
        # derived class: *Wav2Vec2FeatureExtractor*
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )  # Download audio processor from huggingface.co and cache.
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "./test/saved_model/"
        )  # E.g. audio_processor (or model) was saved using *save_pretrained('./test/saved_model/')*
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("./test/saved_model/audio_preprocessor_config.json")
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False
        )
        assert audio_processor.return_attention_mask is False
        audio_processor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False, return_unused_kwargs=True
        )
        assert audio_processor.return_attention_mask is False
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        audio_processor_dict, kwargs = cls.get_audio_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(audio_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save an audio processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~audio_processing_utils.AudioProcessorBase.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the audio processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
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

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_audio_processor_file = os.path.join(save_directory, AUDIO_PROCESSOR_NAME)

        self.to_json_file(output_audio_processor_file)
        logger.info(f"Audio processor saved in {output_audio_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_audio_processor_file]

    @classmethod
    def get_audio_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating an
        audio processor of type [`~audio_processing_utils.AudioProcessorBase`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the audio processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", None)
        token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "audio processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)

        if os.path.isfile(pretrained_model_name_or_path):
            resolved_audio_processor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            audio_processor_file = pretrained_model_name_or_path
            resolved_audio_processor_file = download_url(pretrained_model_name_or_path)
        else:
            try:
                # Try to Load from AUDIO_PROCESSOR_NAME
                audio_processor_file = AUDIO_PROCESSOR_NAME
                resolved_audio_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    AUDIO_PROCESSOR_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                )
            except EnvironmentError:
                # Otherwise fallback to FEATURE_EXTRACTOR_NAME for BC
                audio_processor_file = FEATURE_EXTRACTOR_NAME
                resolved_audio_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    FEATURE_EXTRACTOR_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                )
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load audio processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {AUDIO_PROCESSOR_NAME} file."
                )

        try:
            with open(resolved_audio_processor_file, encoding="utf-8") as reader:
                text = reader.read()
            audio_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise OSError(
                f"It looks like the config file at '{resolved_audio_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_audio_processor_file}")
        else:
            logger.info(
                f"loading configuration file {audio_processor_file} from cache at {resolved_audio_processor_file}"
            )

        if not is_local:
            if "auto_map" in audio_processor_dict:
                audio_processor_dict["auto_map"] = add_model_info_to_auto_map(
                    audio_processor_dict["auto_map"], pretrained_model_name_or_path
                )
            if "custom_pipelines" in audio_processor_dict:
                audio_processor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                    audio_processor_dict["custom_pipelines"], pretrained_model_name_or_path
                )

        return audio_processor_dict, kwargs

    @classmethod
    def from_dict(cls, audio_processor_dict: dict[str, Any], **kwargs) -> "AudioProcessorBase":
        """
        Instantiates a type of [`~audio_processing_utils.AudioProcessorBase`] from a Python dictionary of
        parameters.

        Args:
            audio_processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the audio processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~audio_processing_utils.AudioProcessorBase.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the audio processor object.

        Returns:
            [`~audio_processing_utils.AudioProcessorBase`]: The audio processor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Update audio_processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if key in audio_processor_dict:
                audio_processor_dict[key] = value
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        audio_processor = cls(**audio_processor_dict)

        logger.info(f"Audio processor {audio_processor}")
        if return_unused_kwargs:
            return audio_processor, kwargs
        else:
            return audio_processor

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["audio_processor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "window" in output:
            del output["window"]
        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "AudioProcessorBase":
        """
        Instantiates an audio processor of type [`~audio_processing_utils.AudioProcessorBase`] from the path to
        a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            An audio processor of type [`~audio_processing_utils.AudioProcessorBase`]: The audio processor
            object instantiated from that JSON file.
        """
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        audio_processor_dict = json.loads(text)
        return cls(**audio_processor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this audio processor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this audio processor's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoAudioProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom audio processors as the ones
        in the library are already mapped with `AutoAudioProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoAudioProcessor"`):
                The auto class to register this new audio processor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


AudioProcessorBase.push_to_hub = copy_func(AudioProcessorBase.push_to_hub)
if AudioProcessorBase.push_to_hub.__doc__ is not None:
    AudioProcessorBase.push_to_hub.__doc__ = AudioProcessorBase.push_to_hub.__doc__.format(
        object="audio processor", object_class="AutoAudioProcessor", object_files="audio processor file"
    )
