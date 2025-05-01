from transformers import AutoModel, AutoModelForImageTextToText, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

model_id = "Salesforce/instructblip-flan-t5-xl"
key_mapping = {"language_model.model": "language_model"}
model = AutoModel.from_pretrained(model_id)
print(type(model))

key_mapping = {
    "language_model.model" : "model.language_model",
    "vision_tower": "model.vision_tower",
    "multi_modal_projector" : "model.multi_modal_projector",
    "language_model.lm_head": "lm_head",
}
model = AutoModelForImageTextToText.from_pretrained(model_id)
print(type(model))

