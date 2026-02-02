import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

print("�� Loading tokenizer...")
tokenizer = MistralTokenizer.from_hf_hub(model_id)

print("🧩 Loading model (this may take several minutes)...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 自动分配 GPU
    cache_dir="/blue/sgao1/ji757406.ucf/hf_cache"
)

prompt = "Explain why the sky is blue."
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

print("⚙️ Generating output...")
outputs = model.generate(inputs, max_new_tokens=100)
print("\n✅ Model output:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
