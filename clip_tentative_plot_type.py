from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP (cached already, so this is fast)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("bar_graph_example_5bars.png")

# Encode image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# Normalize
image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Define plot-type hypotheses
plot_types = [
    "a bar chart",
    "a line chart",
    "a scatter plot",
    "a histogram",
    "a pie chart",
    "a heatmap",
    "a box plot",
    "a violin plot",
    "an area chart",
    "a radar chart"
]

# Encode text
text_inputs = processor(
    text=plot_types,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Similarity = evidence
similarity = image_features @ text_features.T

# Print results
for label, score in zip(plot_types, similarity[0]):
    print(f"{label:15s}  {float(score):.3f}")

# similarity shape: [1, num_labels]
scores = similarity[0]

max_idx = scores.argmax().item()
max_label = plot_types[max_idx]
max_score = float(scores[max_idx])

print("\nBest match:")
print(f"{max_label}  {max_score:.3f}")
