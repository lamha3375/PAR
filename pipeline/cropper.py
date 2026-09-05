
from typing import Optional

import cv2
import numpy as np

from .config import (
    CROP_PADDING,
    MIN_CROP_WIDTH,
    MIN_CROP_HEIGHT
)

class PersonCropper:
    def __init__(
        self,
        padding: float = CROP_PADDING,
        min_width: int = MIN_CROP_WIDTH,
        min_height: int = MIN_CROP_HEIGHT
    ):

        self.padding = padding

        self.min_width = min_width

        self.min_height = min_height

    def crop(
        self,
        frame: np.ndarray,
        bbox: list[int]
    ) -> Optional[np.ndarray]:

        if frame is None:
            return None

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        # Bounding box width / height
        box_width = x2 - x1
        box_height = y2 - y1

        if (
            box_width < self.min_width
            or box_height < self.min_height
        ):
            return None

        # Padding
        pad_x = int(
            box_width * self.padding
        )

        pad_y = int(
            box_height * self.padding
        )

        x1 = x1 - pad_x
        y1 = y1 - pad_y

        x2 = x2 + pad_x
        y2 = y2 + pad_y

        # Clamp vào frame
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Crop
        crop = frame[
            y1:y2,
            x1:x2
        ].copy()

        return crop