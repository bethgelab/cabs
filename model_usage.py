import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:bethgelab/ViT-B-32_cabs-dm_recap_0.8")
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")

image = preprocess(Image.open("<image>.jpg")).unsqueeze(0)
text = tokenizer(["<list_of_texts>"])

with torch.no_grad(), torch.cuda.amp.autocast():
 image_features = model.encode_image(image)
 text_features = model.encode_text(text)
 image_features /= image_features.norm(dim=-1, keepdim=True)
 text_features /= text_features.norm(dim=-1, keepdim=True)

 text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)