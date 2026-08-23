from abc import ABC, abstractmethod

import numpy as np


class AttributePredictorInterface(ABC):
    """
    Interface chuẩn giữa TV1 và TV2.

    """

    @abstractmethod
    def predict(
        self,
        crop_image: np.ndarray
    ) -> dict:
        """
        Input:
            crop_image: ảnh người dạng BGR

        Output:
            dict attributes
        """
        pass