# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma4 model."""

import tempfile
import unittest
from contextlib import contextmanager

import pytest
from parameterized import parameterized

from transformers import (
    AutoTokenizer,
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_deterministic_for_accelerator,
    require_deterministic_for_xpu,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import ModelOutput

from ...alm_tester import ALMModelTest, ALMModelTester
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        Gemma4ForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Model,
        Gemma4Processor,
        Gemma4TextModel,
    )
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_vision_model


GEMMA4_RANDOM_MOE_FA2_SKIP_REASON = (
    "Randomly initialized Gemma4 MoE routers are too sensitive to tiny eager/FA2 input differences"
)


class Gemma4ModelTester(VLMModelTester, ALMModelTester):
    base_model_class = Gemma4Model
    config_class = Gemma4Config
    text_config_class = Gemma4TextConfig
    vision_config_class = Gemma4VisionConfig
    audio_config_class = Gemma4AudioConfig
    conditional_generation_class = Gemma4ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("num_hidden_layers", 4)
        kwargs.setdefault("num_kv_shared_layers", 2)
        kwargs.setdefault(
            "layer_types",
            [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        kwargs.setdefault("vocab_size_per_layer_input", 99)
        kwargs.setdefault("hidden_size_per_layer_input", 16)
        kwargs.setdefault("enable_moe_block", True)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("top_k_experts", 2)
        kwargs.setdefault("use_bidirectional_attention", "vision")

        # Clipped linears register inf/-inf buffers which cause NaN in test_torch_save_load's
        # comparison logic (inf - inf = NaN). Disable for testing.
        kwargs.setdefault("use_clipped_linears", False)
        kwargs.setdefault("subsampling_conv_channels", [16, 8])
        kwargs.setdefault("conv_kernel_size", 3)
        kwargs.setdefault("attention_chunk_size", 4)
        kwargs.setdefault("attention_context_left", 5)
        kwargs.setdefault("attention_context_right", 0)
        kwargs.setdefault("output_proj_dims", 32)
        kwargs.setdefault("audio_seq_length", 96)
        kwargs.setdefault("audio_num_channels", 16)

        kwargs.setdefault("image_size", 20)
        kwargs.setdefault("patch_size", 5)
        kwargs.setdefault("pooling_kernel_size", 2)
        kwargs.setdefault("num_image_tokens", 5)
        kwargs.setdefault("num_video_tokens", 25)

        kwargs.setdefault("boi_token_id", 7)
        kwargs.setdefault("eoi_token_id", 8)
        kwargs.setdefault("audio_token_id", 9)
        kwargs.setdefault("boa_token_id", 10)

        kwargs.setdefault("seq_length", 60)

        super().__init__(parent, **kwargs)
        self.per_layer_config = {
            layer_idx: {"head_dim": 2 * self.head_dim}
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        }

    def create_audio_features(self):
        input_features = floats_tensor([self.batch_size, self.audio_seq_length, self.audio_num_channels])
        return input_features

    def create_pixel_values(self):
        # (num_images, max_num_patches, patch_size * patch_size * num_channels)
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.image_size,
                self.patch_size * self.patch_size * self.num_channels,
            ]
        )
        return pixel_values

    def create_pixel_values_videos(self):
        pixel_values_videos = floats_tensor(
            [
                self.batch_size * self.num_frames,
                self.image_size,
                self.patch_size * self.patch_size * self.num_channels,
            ]
        )
        return pixel_values_videos

    def _prepare_image_inputs(self, input_ids, config, modality_inputs):
        input_ids, data = super()._prepare_image_inputs(input_ids, config, modality_inputs)
        mm_token_type_ids = data.get("mm_token_type_ids", torch.zeros_like(input_ids))
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        data["mm_token_type_ids"] = mm_token_type_ids

        # (num_images, max_num_patches, 2) for height/width positions. Let it be all ones for testign
        pixel_position_ids = torch.ones(self.image_size, device=torch_device, dtype=torch.long)
        pixel_position_ids = pixel_position_ids[None, :, None].repeat(self.batch_size, 1, 2)
        # create (h*w, 2) grid of (x, y) coords for a non-square input image
        h = int(self.image_size**0.5)
        w = self.image_size // h
        xs = torch.arange(w).repeat(h)
        ys = torch.arange(h).repeat_interleave(w)
        pixel_position_ids = torch.stack([xs, ys], dim=-1).to(device=torch_device)
        data["image_position_ids"] = pixel_position_ids.unsqueeze(0).repeat(self.batch_size, 1, 1)

        return input_ids, data

    def _prepare_video_inputs(self, input_ids, config, modality_inputs):
        input_ids, data = super()._prepare_video_inputs(input_ids, config, modality_inputs)
        mm_token_type_ids = data.get("mm_token_type_ids", torch.zeros_like(input_ids))
        mm_token_type_ids[input_ids == self.video_token_id] = 2
        data["mm_token_type_ids"] = mm_token_type_ids

        # (num_images, max_num_patches, 2) for height/width positions. Let it be all ones for testign
        pixel_position_ids = torch.ones(self.image_size, device=torch_device, dtype=torch.long)
        pixel_position_ids = pixel_position_ids[None, None, :, None].repeat(self.batch_size, self.num_frames, 1, 2)
        # create (h*w, 2) grid of (x, y) coords for a non-square input image
        h = int(self.image_size**0.5)
        w = self.image_size // h
        xs = torch.arange(w).repeat(h)
        ys = torch.arange(h).repeat_interleave(w)
        pixel_position_ids = torch.stack([xs, ys], dim=-1).to(device=torch_device)
        data["video_position_ids"] = pixel_position_ids[None, None, ...].repeat(self.batch_size, self.num_frames, 1, 1)
        return input_ids, data

    def _prepare_audio_inputs(self, input_ids, config, modality_inputs):
        input_ids, data = super()._prepare_audio_inputs(input_ids, config, modality_inputs)
        mm_token_type_ids = data.get("mm_token_type_ids", torch.zeros_like(input_ids))
        mm_token_type_ids[input_ids == self.audio_token_id] = 3
        data["mm_token_type_ids"] = mm_token_type_ids
        data["input_features_mask"] = torch.ones(
            self.batch_size, self.audio_seq_length, dtype=torch.bool, device=torch_device
        )
        return input_ids, data

    def get_audio_embeds_mask(self, audio_embeds_mask):
        return torch.ones(self.batch_size, self.audio_seq_length // 4, dtype=torch.bool, device=torch_device)


@require_torch
class Gemma4ModelTest(VLMModelTest, ALMModelTest, unittest.TestCase):
    model_tester_class = Gemma4ModelTester
    MODALITY_COMBINATIONS = [("image",), ("video",), ("audio",)]
    # TODO: bring back model-specific tests that were deleted

@slow
@require_torch_accelerator
class Gemma4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "google/gemma-4-E2B-it"
        self.processor = Gemma4Processor.from_pretrained(self.model_name)

        self.url1 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        self.url2 = url_to_local_path(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
        )
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url1},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_deterministic_for_xpu
    def test_model_with_image(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean** in the background under a **clear'],
                ("xpu", 5): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean** in the background under a **clear'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_with_image_batch(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": self.url1,
                    },
                    {"type": "image", "url": self.url2},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background",
                    "No, these images are **not identical**.\n\nHere's a breakdown of the differences:\n\n1.  **Image 1 (Cow on",
                ],
                ("xpu", 5): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean** in the background under a **clear",
                    "No, these images are **not identical**.\n\nHere's a breakdown of the differences:\n\n1.  **Image 1 (Cow on",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_multiimage(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url2},
                    {"type": "text", "text": "What do you see here?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Roadway:** There is an'],
                ("cuda", (9, 0)): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Roadway:** There is an'],
                ("xpu", 5): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Roadway:** There is an'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_multi_gpu
    def test_model_text_only_multigpu(self):
        """Accelerate destroys the input dict `shared_kv_states` if it's not passed as kwarg and part of
        `_skip_keys_device_placement`, so test this to avoid regresions.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA loom of logic, spun from endless thread,\nWhere data streams in, and the patterns spread.\nNo'],
                ("cuda", (9, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_text_only(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA loom of logic, spun from endless thread,\nWhere data streams in, and the patterns spread.\nNo'],
                ("cuda", (9, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("xpu", 5): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_states_sharing_with_and_without_cache(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Who are you? What can you do?"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)
        input_size = inputs.input_ids.shape[-1]

        # With and without cache generatiom should share kv states the same way
        output_with_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=True)
        output_without_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=False)

        output_text_with_cache = tokenizer.batch_decode(output_with_cache[:, input_size:], skip_special_tokens=True)
        output_text_without_cache = tokenizer.batch_decode(
            output_without_cache[:, input_size:], skip_special_tokens=True
        )

        self.assertEqual(output_text_with_cache, output_text_without_cache)

    # Note: we do not test FA2 as the head dim is 512 on some layers, which is not compatible with the kernels
    @parameterized.expand([("sdpa",), ("eager",)])
    @require_deterministic_for_accelerator(devices=["cuda"])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. Outputs for every attention functions
        should be coherent and identical.
        """

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding="left")
        input_text = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in input_text
        ]
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = Gemma4ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=torch_device,
            attn_implementation=attn_implementation,
        )

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.get_text_config().sliding_window)

        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, cache_implementation="static")
        output_text = tokenizer.batch_decode(out[:, input_size:])

        EXPECTED_COMPLETIONS = Expectations(
            {
                ("cuda", 8): [
                    "That sounds lovely! It seems like you're really enjoying the place you'"
                    if attn_implementation == "sdpa"
                    else "That sounds like a very pleasant place! It seems like you're really enjoying",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
                ("xpu", 5): [
                    "That sounds lovely! It seems like you're really enjoying the place you'",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
            }
        )
        self.assertEqual(output_text, EXPECTED_COMPLETIONS.get_expectation())

    @pytest.mark.torch_export_test
    def test_export_text_only(self):
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        # Run on CPU: the full E2B model (~4 GiB bfloat16) + torch.export tracing overhead
        # (~4 GiB) exceeds the 22.3 GiB GPU memory available in CI. CPU avoids the OOM.
        # max_cache_len=19 covers the prompt (~16 tokens) + 3 new tokens with a small buffer.
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model, batch_size=1, max_cache_len=19, device="cpu")
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], device="cpu", dtype=torch.long),
        )

        # Test generation with the exported model
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        max_new_tokens_to_generate = 3
        # Generate text with the exported model
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate, device="cpu"
        )

        input_text = tokenizer(prompt, return_tensors="pt").to("cpu")
        eager_outputs = model.generate(
            **input_text,
            max_new_tokens=max_new_tokens_to_generate,
            do_sample=False,  # Use greedy decoding to match the exported model
        )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        self.assertEqual(export_generated_text, eager_generated_text)
