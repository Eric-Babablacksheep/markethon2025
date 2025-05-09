import os
import logging
import torch
from typing import Optional, Dict, Callable

from model import AlexNet

class TrainingConfig:
    """Training configuration class for managing basic model training parameters"""

    def __init__(
            self,
            model_name: str = 'AlexNet',
            batch_size: int = 1,
            num_classes: int = 12,
            test_data_path: str = '../data/split-data/test',
            image_path: str = '',
            pretrained_weights: Optional[str] = None
    ):
        # Initialize training configuration parameters
        self.model_name = model_name
        self.batch_size = batch_size
        self.test_data_path = test_data_path
        self.num_classes = num_classes
        self.image_path = image_path
        self.pretrained_weights = pretrained_weights
        # Automatically detect and select the available device (GPU/CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Mac M1/M2/M3
        else:
            self.device = torch.device("cpu")  # CPU

class ModelChoose:
    """Model selection and initialization class, supporting multiple deep learning models and handling pretrained weights"""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the model selector

        :param config: Training configuration object
        """
        self.config = config
        self.logger = self._setup_logger()
        # Define a mapping of supported models
        self.supported_models: Dict[str, Callable] = {
            "AlexNet": self._create_alexnet,
        }

    def _create_model_factory(self, model_func: Callable) -> torch.nn.Module:
        """
        Generic model creation factory method

        :param model_func: Model creation function
        :return: Initialized model instance
        """
        return model_func(num_classes=self.config.num_classes).to(self.config.device)
    
    def _create_alexnet(self) -> torch.nn.Module:
        return AlexNet(num_classes=self.config.num_classes, init_weights=True).to(self.config.device)

    def initialize_model(self) -> torch.nn.Module:
        """
        Initialize the model and load pretrained weights
        """
        # Retrieve the model creation function from the mapping dictionary
        model_creator = self.supported_models.get(
            self.config.model_name,
            self._create_alexnet  
        )
        model = model_creator()
        state = torch.load(
            self.config.pretrained_weights,
            map_location=self.config.device
        )
        
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model = state

        return model

    def _setup_logger(self):
        """
        Set up the logger

        :return: Configured logger
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)