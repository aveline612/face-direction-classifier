# src/inference.py - Production-ready inference
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from typing import Dict, Union, Optional
import json
import time
import os

# Configure logging
logger = logging.getLogger(__name__)


class FaceDirectionClassifier:
    """Production-ready face direction classifier"""

    def __init__(self, model_path: str = "models/face_direction_classifier.pth"):
        self.model_path = model_path
        self.device = self._get_device()
        self.transform = self._get_transforms()
        self.class_names = ['back', 'front', 'side']
        self.model = None
        self.load_model()

    def _get_device(self):
        """Get the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _get_transforms(self):
        """Define image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """Load the model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at: {self.model_path}")

            # Create model architecture
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))

            # Load weights
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            self.model = model
            logger.info(f"Model loaded successfully from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model = None
            raise

    def predict(self, image: Union[str, Image.Image, np.ndarray],
                return_all: bool = True) -> Dict:
        """
        Predict face direction with production-grade error handling

        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_all: Whether to return all probabilities

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        try:
            # Validate model
            if self.model is None:
                raise RuntimeError("Model not loaded")

            # Load and validate image
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                raise ValueError("Image must be file path, PIL Image, or numpy array")

            # Validate image dimensions
            if pil_image.size[0] < 50 or pil_image.size[1] < 50:
                raise ValueError("Image too small (min 50x50 pixels)")

            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(outputs, 1).item()
                confidence = probabilities[0][predicted_idx].item()

            predicted_class = self.class_names[predicted_idx]
            inference_time = time.time() - start_time

            # Build result
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence, 4),
                'inference_time_ms': round(inference_time * 1000, 2),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model_version': '1.0.0'
            }

            if return_all:
                result['probabilities'] = {
                    cls: round(prob.item(), 4)
                    for cls, prob in zip(self.class_names, probabilities[0])
                }

            logger.info(f"Prediction: {predicted_class} ({confidence:.2%}) in {inference_time:.3f}s")
            return result

        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'inference_time_ms': round(inference_time * 1000, 2),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def batch_predict(self, images: list, batch_size: int = 8) -> list:
        """Batch prediction for efficiency"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                results.append(self.predict(img, return_all=False))
        return results

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'device': str(self.device),
            'classes': self.class_names,
            'input_size': (224, 224),
            'architecture': 'ResNet-18',
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }


# Singleton instance for production use
_classifier_instance = None


def get_classifier():
    """Get or create classifier instance (singleton pattern)"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FaceDirectionClassifier()
    return _classifier_instance


def predict_image(image_path: str) -> Dict:
    """Convenience function for quick predictions"""
    classifier = get_classifier()
    return classifier.predict(image_path)