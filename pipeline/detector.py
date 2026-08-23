
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO

from .config import (
    YOLO_MODEL_PATH,
    PERSON_CLASS_ID,
    DETECTION_CONFIDENCE
)


class PersonDetector:
    """
    YOLOv8 Person Detector.

    Nhiệm vụ:
        - Load YOLO model
        - Detect person
        - Trả về bounding box
        - Trả về confidence

    Không chịu trách nhiệm tracking.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        confidence: float = DETECTION_CONFIDENCE,
        device: str | None = None
    ):

        self.model_path = model_path

        self.confidence = confidence

        self.device = device

        # Load YOLO
        self.model = YOLO(model_path)

    def detect(
        self,
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect người trong một frame.

        Args:
            frame:
                OpenCV BGR frame.

        Returns:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.92,
                    "class_id": 0
                }
            ]
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        if len(result.boxes) == 0:
            return detections

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        class_ids = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for bbox, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids
        ):

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            detections.append(
                {
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "confidence": float(
                        confidence
                    ),

                    "class_id": int(
                        class_id
                    )
                }
            )

        return detections