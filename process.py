from transformers import AutoProcessor, AutoModelForImageTextToText, AutoImageProcessor
import torch
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity

image = Image.open("/raid/raushan/image.png")
prompt = "USER <image> any text here. ASSISTANT:"
batch_size = 100

pil_images = [image]*batch_size
tensor_image = torch.randint(0, 256, size=(3, 256, 256), dtype=torch.uint8)
batched_images = torch.randint(0, 256, size=(batch_size, 3, 256, 256), dtype=torch.uint8)

vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# warm up
for _ in range(3):
    processed_inputs = vit_processor(
        images=batched_images.to("cuda"),
        return_tensors="pt"
    )

batched_images = torch.randint(0, 256, size=(batch_size, 3, 256, 256), dtype=torch.uint8)

start.record()
processed_inputs = vit_processor(
    images=batched_images.to("cuda"),
    return_tensors="pt"
)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("processor"):
        processed_inputs = vit_processor(
            images=batched_images.to("cuda"),
            return_tensors="pt"
        )
    

# processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
# model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device_map="cuda:0", torch_dtype="float16")
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("processor"):
#         processed_inputs = processor(
#             text=[prompt]*batch_size, images=[image]*batch_size,
#             return_tensors="pt", padding=True, do_rescale=False,
#         ).to(model.device, torch.float16)
#     
#     with record_function("model"):
#         output = model.generate(
#             **processed_inputs, max_new_tokens=10
#         )
    
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
