import numpy as np
from PIL import Image
import torch
from inference import load_stage2_encoder
from training.hera_model import create_image_transform

checkpoint_path = "runs/stage2/last.pt"  
image_path = "data/images/example.png"   
output_path = "image_feature.npy"        
modality = "radiology"                   
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use the image normalization parameters stored in the checkpoint's training configuration.
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
data_config = checkpoint["config"].get("data", {})
mean = data_config.get("mean", [0.5, 0.5, 0.5])
std = data_config.get("std", [0.5, 0.5, 0.5])
del checkpoint

encoder = load_stage2_encoder(checkpoint_path, device=device)
size = encoder.model.config.image_size
transform = create_image_transform(size, mean=mean, std=std)
with Image.open(image_path) as source:
    images = transform(source).unsqueeze(0).to(device)

feature = encoder.encode_image(images, modality)  # [1, projection_dim]
np.save(output_path, feature[0].cpu().numpy())
print("Saved:", output_path, "shape:", tuple(feature.shape))
