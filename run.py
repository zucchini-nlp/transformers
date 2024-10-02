from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print(type(tokenizer), tokenizer.bos_token, tokenizer.special_tokens_map)

print(f"\n{'-' * 100}\n")

tokenizer_vlm = AutoTokenizer.from_pretrained("/raid/raushan/tokenizer_vlm")
tokenizer_vlm.eoi_token = "<s>"
print(type(tokenizer_vlm), tokenizer_vlm.eoi_token_id, tokenizer_vlm.special_tokens_map)

print(tokenizer_vlm.eoi_token, tokenizer_vlm.eoi_token_id)
print(tokenizer_vlm.special_tokens_map)
print(tokenizer_vlm("HELLO GUYS"))

# tokenizer_vlm.save_pretrained("/raid/raushan/tokenizer_vlm")

print(f"\n{'-' * 100}\n")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print(type(tokenizer), tokenizer.SPECIAL_TOKENS_ATTRIBUTES)

print(f"\n{'-' * 100}\n")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
# print(type(tokenizer))
