from typing import Dict
from PIL import Image
import json

# Assuming these are your own modules, make sure they are implemented correctly
from model_choose import ModelChoose, TrainingConfig
from predictions import Prediction

class PredictionService:
    """
    Service class to handle predictions on images using different deep learning models.
    """

    # Available models mapping to pretrained weights paths
    AVAILABLE_MODELS = {
        "AlexNet": "./weights/AlexNet_model_92.04%.pth",
    }

    @staticmethod
    def is_model_available(model_name: str) -> bool:
        """
        Check if the requested model is available and has weights file.
        
        Args:
            model_name (str): Name of the model.

        Returns:
            bool: True if model is available, False otherwise.
        """
        return model_name in PredictionService.AVAILABLE_MODELS and \
               PredictionService.AVAILABLE_MODELS[model_name] is not None

    @staticmethod
    def predict_image(image: Image.Image, model_name: str = "AlexNet") -> Dict:
        """
        Predict the class of an image using the specified model.
        
        Args:
            image (PIL.Image.Image): Input image.
            model_name (str): Name of the model to use.

        Returns:
            Dict: A dictionary containing prediction results.
        """
        # Step 3: Check if model is available
        if not PredictionService.is_model_available(model_name):
            raise ValueError(f"Model '{model_name}' is not available or pretrained weights are missing.")

        # Step 4: Set up the training configuration
        config = TrainingConfig(
            model_name=model_name,
            num_classes=12,  # Adjust if your task has a different number of classes
            pretrained_weights=PredictionService.AVAILABLE_MODELS[model_name]
        )

        # Step 5: Initialize the model
        model_selector = ModelChoose(config)
        model = model_selector.initialize_model()

        # Step 6: Run prediction
        predictor = Prediction(config, model, logger=model_selector.logger)
        result_json = predictor.run(image)

        # Step 7: Return result as a dictionary
        return json.loads(result_json)

