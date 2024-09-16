from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# print(type(tokenizer), tokenizer._auto_class)


tokenizer_vlm = AutoTokenizer.from_pretrained("/raid/raushan/tokenizer_vlm")
tokenizer_vlm.eoi_token = "<s>"
print(type(tokenizer_vlm), tokenizer_vlm.eoi_token_id, tokenizer_vlm.special_tokens_map)

# print(tokenizer_vlm.eoi_token, tokenizer_vlm.eoi_token_id, tokenizer_vlm.base_tokenizer._auto_class)
# print(tokenizer_vlm.special_tokens_map, tokenizer_vlm.special_tokens_map_extended)
tokenizer_vlm.save_pretrained("/raid/raushan/tokenizer_vlm")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# print(type(tokenizer), tokenizer.SPECIAL_TOKENS_ATTRIBUTES)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
# print(type(tokenizer))
