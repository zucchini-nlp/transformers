import os
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
from transformers import LlavaForConditionalGeneration, LlavaConfig, LlavaProcessor
from typing import Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

clip_model_name_or_path = "OpenGVLab/InternViT-300M-448px-V2_5"
qwen_model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
phi_3_vision = "microsoft/Phi-3-vision-128k-instruct"

# vision_config = AutoConfig.from_pretrained(phi_3_vision, trust_remote_code='True')
# text_config = AutoConfig.from_pretrained(qwen_model_name_or_path) 
# configuration = LlavaConfig(vision_config, text_config)
# print(configuration)

# model = LlavaForConditionalGeneration(configuration, trust_remote_code=True)
# model.save_pretrained("model001")

# image_processor = AutoImageProcessor.from_pretrained(clip_model_name_or_path, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)
# processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer, patch_size=14, vision_feature_select_strategy="default")
# processor.save_pretrained("model001")

# -----------------------------------------------------------------------------------------------------

model_name_or_path = "model001"

llava_processor = LlavaProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write me a novel about dragons."},
]

prompt = llava_processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = llava_processor(text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=20)
gen_text = llava_processor.batch_decode(
    generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

print(gen_text)

