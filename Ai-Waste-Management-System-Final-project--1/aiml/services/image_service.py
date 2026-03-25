import torch
import torch.nn.functional as F
from pathlib import Path
from aiml.utils.logger import get_logger
from aiml.utils.preprocessing import preprocess_image
from aiml.models.waste_classifier.model import load_model
import yaml

logger = get_logger(__name__)

# Load config
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        cv_config = config.get("model", {}).get("cv", {})
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    cv_config = {"num_classes": 4, "classes": ["wet", "dry", "plastic", "hazardous"], "image_size": 224}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageClassificationService:
    def __init__(self):
        logger.info(f"Initializing ImageClassificationService on device: {DEVICE}")
        self.classes = cv_config.get("classes", ["wet", "dry", "plastic", "hazardous"])
        self.num_classes = cv_config.get("num_classes", len(self.classes))
        self.image_size = cv_config.get("image_size", 224)
        model_path_str = cv_config.get("model_path", "models/waste_classifier/mobilenetv2_waste.pth")
        
        # Resolve absolute path for model
        base_dir = Path(__file__).parent.parent
        self.model_path = base_dir / model_path_str
        
        # Load model lazy or at init
        self.model = load_model(str(self.model_path), self.num_classes, DEVICE)

    def classify_image(self, image_bytes: bytes) -> dict:
        """Classifies an image and returns the predicted class and confidence."""
        try:
            tensor = preprocess_image(image_bytes, self.image_size).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.classes[predicted_idx.item()]
            
            result = {
                "waste_type": predicted_class,
                "confidence": round(confidence.item(), 4)
            }
            logger.info(f"Image classified as {predicted_class} with {confidence.item():.2f} confidence.")
            return result
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise e

image_service = ImageClassificationService()
