import torch
from torchvision import transforms
from PIL import Image
import io

def get_image_transforms(image_size: int = 224):
    """Returns PyTorch vision transforms to standardize input images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_bytes: bytes, image_size: int = 224) -> torch.Tensor:
    """Preprocess raw image bytes to a tensor suitable for PyTorch model inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_image_transforms(image_size)
    tensor = transform(image)
    # Add batch dimension to create a (1, C, H, W) tensor
    return tensor.unsqueeze(0)
