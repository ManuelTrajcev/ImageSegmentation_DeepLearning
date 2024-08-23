import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from model import UNET
import os
from utils import decode_segmap, label_colours

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET(in_channels=3, out_channels=20).to(DEVICE)
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Preprocess the image
val_transforms = A.Compose(
    [
        A.Resize(height=160, width=240),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    transformed = val_transforms(image=image)

    return transformed["image"].unsqueeze(0)

input_image = preprocess_image("images/tmp2.png").to(DEVICE)

# Perform inference
with torch.no_grad():
    output = model(input_image)
    output = torch.sigmoid(output)
    output = (output > 0.5).float()

# Post-process and visualize the output
output_image = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Convert to class indices
output_image = decode_segmap(output_image, label_colours)

print(output_image)
plt.imshow(output_image)
plt.show()

# Save the segmented image
output_image_pil = Image.fromarray(output_image)
output_image_pil.save("segmented_images/segmented_output.png")
