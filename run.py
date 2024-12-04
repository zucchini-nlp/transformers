from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.cuda.set_per_process_memory_fraction(0.12, 0)
torch.cuda.set_per_process_memory_fraction(0.12, 1)

model_id = "cuda:0-llama/cuda:0-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("A very long inputs text here", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=2300, cache_implementation="offloaded_static")
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0][:100])
