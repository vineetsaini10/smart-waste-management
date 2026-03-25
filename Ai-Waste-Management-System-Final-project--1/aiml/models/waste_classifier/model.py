import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class WasteClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(WasteClassifier, self).__init__()
        # Load pretrained MobileNetV2
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Freeze early layers if we want, but for now we'll just replace the classifier head
        # The classifier of MobileNetV2 is a Sequential with Dropout and Linear
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def load_model(model_path: str, num_classes: int = 4, device: str = "cpu") -> nn.Module:
    """Loads a trained model from disk."""
    model = WasteClassifier(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Using untrained weights for demonstration.")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
    
    model.to(device)
    model.eval()
    return model
