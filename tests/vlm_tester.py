# Copyright 2026 HuggingFace Inc.
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
import pytest
import functools
import inspect
import unittest
from inspect import signature

from .multimodal_tester import MultiModalModelTest, MultiModalModelTester, ids_tensor
from .test_modeling_common import (
    floats_tensor,
    is_torch_available,
    torch_device,
)


if is_torch_available():
    import torch


class VLMModelTester(MultiModalModelTester):
    vision_config_class = None
    _required_attributes = MultiModalModelTester._required_attributes + ("base_model_class", "vision_config_class")

    @property
    def pipeline_model_mapping(self):
        return {
            "feature-extraction": self.base_model_class,
            "image-text-to-text": self.conditional_generation_class,
        }

    def __init__(self, parent, **kwargs):
        # Overrides of _TEXT_MODEL_TESTER_DEFAULTS
        kwargs.setdefault(
            "seq_length",
            7
            + kwargs.get(
                "num_image_tokens",
                (kwargs.get("image_size", 8) // kwargs.get("patch_size", 4)) ** 2,
            )
            + kwargs.get(
                "num_video_tokens",
                kwargs.get("num_frames", 4) * (kwargs.get("image_size", 8) // kwargs.get("patch_size", 4)) ** 2,
            ),
        )
        kwargs.setdefault("pad_token_id", 0)

        # VLM-specific defaults
        kwargs.setdefault("use_token_type_ids", False)
        kwargs.setdefault("hidden_dropout_prob", 0.1)
        kwargs.setdefault("attention_probs_dropout_prob", 0.1)
        kwargs.setdefault("type_vocab_size", 16)
        kwargs.setdefault("type_sequence_label_size", 2)
        kwargs.setdefault("initializer_range", 0.02)
        kwargs.setdefault("num_labels", 3)
        kwargs.setdefault("num_choices", 4)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("is_decoder", False)
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("num_frames", 4)
        kwargs.setdefault("projection_dim", 32)
        kwargs.setdefault("projector_hidden_act", "gelu")
        kwargs.setdefault("vision_feature_select_strategy", "default")
        kwargs.setdefault("vision_feature_layer", -1)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("num_image_tokens", (kwargs["image_size"] // kwargs["patch_size"]) ** 2)
        kwargs.setdefault(
            "num_video_tokens", (kwargs["image_size"] // kwargs["patch_size"]) ** 2 * kwargs["num_frames"]
        )

        super().__init__(parent, **kwargs)

        # Computed default depending on base-class defaults for hidden_size / num_attention_heads.
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_attention_heads

    # allow inputs to be prepared for each supported vision modality and its combinations
    def prepare_config_and_inputs_for_common(self, modalities: list[str] | None = None):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # Avoid flaky tests by scrubbing any accidental special tokens produced by ids_tensor.
        # Modality placeholder tokens are scrubbed and placed by `_prepare_modality_inputs`.
        safe_token_id = self._safe_token_id()
        for token_id in self._special_token_ids:
            input_ids[input_ids == token_id] = safe_token_id

        # Create attention mask with final input_ids (after modality placeholders are placed) — important
        # for models that derive padding from token values.
        attention_mask = self.create_attention_mask(input_ids) if self.use_input_mask else None
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        if modalities is not None:
            modality_inputs = {}
            for modality in modalities:
                input_ids, current_data = self._prepare_modality_inputs(input_ids, config, modality=modality)
                current_data.update(self.get_additional_inputs(config, input_ids, current_data, modality=modality))
                modality_inputs.update(current_data)
            inputs_dict.update(modality_inputs)
            inputs_dict["input_ids"] = input_ids  # re-set to add placeholder IDs
        return config, inputs_dict

    # -- Overridable VLM-specific hooks ------------------------------------------------------

    def create_pixel_values(self):
        # Override to 5D for patch-based models
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size], scale=1.0)

    def create_pixel_values_videos(self):
        # Override for patch-based models
        return floats_tensor(
            [self.batch_size, self.num_frames, self.num_channels, self.image_size, self.image_size], scale=1.0
        )

    def place_image_tokens(self, input_ids, config):
        # Override if the image tokens shouldn't be placed at the start of the test sequence
        image_token_id = getattr(config, "image_token_id", self.image_token_id)
        # Clear any accidental image tokens first
        input_ids = input_ids.clone()
        input_ids[input_ids == image_token_id] = self.bos_token_id
        # Place image tokens at the start
        input_ids[:, : self.num_image_tokens] = image_token_id
        return input_ids

    def place_video_tokens(self, input_ids, config):
        # Override if the video tokens shouldn't be placed at the start of the test sequence
        video_token_id = getattr(config, "video_token_id", self.video_token_id)
        # Clear any accidental video tokens first
        input_ids = input_ids.clone()
        input_ids[input_ids == video_token_id] = self.bos_token_id
        # Place video tokens after image
        input_ids[:, self.num_image_tokens : self.num_image_tokens + self.num_video_tokens] = video_token_id
        return input_ids

    # -- Hooks consumed by the shared base ---------------------------------------------------

    @property
    def _special_token_ids(self):
        special_tokens = super()._special_token_ids | {self.image_token_id}
        if "video" in self.base_model_class.input_modalities:
            special_tokens |= {self.video_token_id}
        return special_tokens

    def _build_modality_sub_configs(self):
        return {"vision_config": self.get_vision_config()}

    def _prepare_modality_inputs(self, input_ids, config, modality: str):
        data = {}
        if modality == "image":
            data["pixel_values"] = self.create_pixel_values()
            input_ids = self.place_image_tokens(input_ids, config)
        elif modality == "video":
            data["pixel_values_videos"] = self.create_pixel_values_videos()
            input_ids = self.place_video_tokens(input_ids, config)
        else:
            raise ValueError(f"Unrecognized modality={modality}")
        return input_ids, data

    # -- Vision sub-config construction ------------------------------------------------------

    @property
    def vision_config_args(self):
        return list(signature(self.vision_config_class.__init__).parameters.keys())

    def get_vision_config(self):
        kwargs = self._collect_kwargs(self.vision_config_args, self.vision_config_class)
        return self.vision_config_class(**kwargs)


class VLMModelTest(MultiModalModelTest):
    """
    Base test class for Vision-Language Models.

    Subclasses should set:
    - `model_tester_class`: The tester class (subclass of VLMModelTester)

    Optional:
    - `all_model_classes`: Override if not using default from model_tester
    - `pipeline_model_mapping`: Override if not using default from model_tester
    """

    # DON'T set `current_modalities` in models! It's auti-set when initializing a subclass
    current_modalities = None
    MODALITY_COMBINATIONS = [("image",), ("video",), ("image", "video")]

    # All `test_xxx` NOT listed here is assumed to depend on
    # `prepare_config_and_inputs_for_common()` and gets fanned out per modality
    # Do not change it per model test as well!
    MODALITY_INDEPENDENT_TESTS = {
        "test_config",
        "test_model_is_small",
        "test_from_pretrained_no_checkpoint",
        "test_keep_in_fp32_modules_exist",
        "test_keep_in_fp32_modules",
        "test_save_load_keys_to_ignore_on_save",
        "test_load_contiguous_weights",
        "test_can_init_all_missing_weights",
        "test_init_weights_can_init_buffers",
        "test_all_tensors_are_parameter_or_buffer",
        "test_resize_tokens_embeddings",
        "test_model_get_set_embeddings",
        "test_model_main_input_name",
        "test_model_base_model_prefix",
        "test_correct_missing_keys",
        "test_can_use_safetensors",
        "test_load_save_without_tied_weights",
        "test_tied_weights_keys",
        "test_model_weights_reload_no_missing_tied_weights",
        "test_disk_offload_bin",
        "test_disk_offload_safetensors",
        "test_cpu_offload",
        "test_load_with_mismatched_shapes",
        "test_can_load_ignoring_mismatched_shapes",
        "test_attn_implementation_composite_models",
        "test_sdpa_can_dispatch_composite_models",
        "test_generation_tester_mixin_inheritance",
        "test_can_be_initialized_on_meta",
        "test_can_load_with_device_context_manager",
        "test_can_load_with_global_device_set",
        "test_cannot_load_with_meta_device_context_manager",
        "test_config_attn_implementation_setter",
        "test_internal_model_config_and_subconfig_are_same",
        "test_can_set_attention_dynamically",
        "test_can_set_attention_dynamically_composite_model",
        "test_bc_torch_dtype",
        "test_tp_plan_matches_params",
        "test_reverse_loading_mapping",
        "test_can_load_from_already_mapped_keys",
        "test_format_of_can_record_outputs",
        "test_can_capture_specific_layers_hidden_states",
        "test_kernels_can_load_without_crashing",
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "model_tester_class" not in cls.__dict__:
            return

        supported = cls.model_tester_class.base_model_class.input_modalities
        combos = [c for c in cls.MODALITY_COMBINATIONS if all(m in supported for m in c)]
        test_names = [
            name
            for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if name.startswith("test_") and name not in cls.MODALITY_INDEPENDENT_TESTS
        ]

        for name in test_names:
            original = getattr(cls, name)

            for combo in combos:
                new_name = f"{name}_{'_'.join(combo)}"

                @functools.wraps(original)
                def wrapper(self, *args, __orig=original, __combo=combo, **kw):
                    self.current_modalities = __combo
                    return __orig(self, *args, **kw)
                setattr(cls, new_name, wrapper)

                for modality in combo:
                    wrapper = getattr(pytest.mark, modality)(wrapper)
                wrapper = pytest.mark.multimodal_combo("_".join(combo))(wrapper)

    def prepare_config_and_inputs_for_common(self):
        return self.model_tester.prepare_config_and_inputs_for_common(self.current_modalities)

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompt has multiple image tokens.
        """
        if self.current_modalities is None or "video" in self.current_modalities:
            self.skipTest("just skip for now and make a proper test to test current modaity tokens")

        config, input_dict = self.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # Test 1: remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = curr_input_dict["image_sizes"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 2: simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            # First, take just the first item from each tensor
            curr_input_dict = {key: val[:1] for key, val in curr_input_dict.items()}

            # Double the batch size for all batch-dimension tensors except pixel_values
            # This simulates having 2 prompts (each with image tokens) but only 1 image
            batch_tensors_to_double = ["input_ids", "attention_mask", "token_type_ids"]
            for key in batch_tensors_to_double:
                if key in curr_input_dict and curr_input_dict[key] is not None:
                    curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 3: two images and two image tokens don't raise an error
            curr_input_dict["pixel_values"] = torch.cat(
                [curr_input_dict["pixel_values"], curr_input_dict["pixel_values"]], dim=0
            )
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = torch.cat(
                    [curr_input_dict["image_sizes"], curr_input_dict["image_sizes"]], dim=0
                )
            _ = model(**curr_input_dict)

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. "
        "Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass
