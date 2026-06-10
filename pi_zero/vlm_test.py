import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from transformers.image_utils import load_image

model_id = "google/paligemma2-3b-pt-224"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image(url)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="./models/paligemma2-3b-pt-224/",
).eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Leaving the prompt blank for pre-trained models
prompt = ""
model_inputs = (
    processor(text=prompt, images=image, return_tensors="pt")
    .to(torch.bfloat16)
    .to(model.device)
)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
