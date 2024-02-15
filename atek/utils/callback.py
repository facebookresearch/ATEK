from abc import ABC, abstractmethod


class BaseCallback(ABC):
    """
    Base callback class for model inference pipeline
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def __call__(self):
        raise NotImplementedError
