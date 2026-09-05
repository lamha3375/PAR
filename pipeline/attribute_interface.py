from abc import ABC, abstractmethod

import numpy as np


class AttributePredictorInterface(ABC):

    @abstractmethod
    def predict(
        self,
        crop_image: np.ndarray
    ) -> dict:
        pass