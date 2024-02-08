from abc import ABC, abstractmethod
from typing import Dict


class AtekViewer(ABC):
    """
    Abstract class to visualize different model input and predictions
    """

    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): dict-based configurations for the viewer
        """
        self.config = config

    @abstractmethod
    def __call__(self):
        raise NotImplementedError
