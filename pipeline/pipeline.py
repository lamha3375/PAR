import time
from typing import Union

import cv2
import numpy as np

from .video_reader import VideoReader
from .detector import PersonDetector
from .tracker import PersonTracker
from .cropper import PersonCropper
from .smoother import AttributeSmoother
from .attribute_interface import AttributePredictorInterface

from .config import (
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE,
    WINDOW_NAME,
    BOX_THICKNESS,
    FONT_SCALE,
    FONT_THICKNESS,
    SMOOTHING_WINDOW_SIZE,
    SMOOTHING_MAX_MISSING_FRAMES,
    SHOW_FPS,
    SHOW_PERSON_COUNT,
    SHOW_TRACK_ID,
    SHOW_CONFIDENCE
)


class PersonPipeline:

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        confidence: float = DETECTION_CONFIDENCE,
        device: str | None = None,
        attribute_predictor: AttributePredictorInterface | None = None
    ):

        self.detector = PersonDetector(
            model_path=model_path,
            confidence=confidence,
            device=device
        )

        self.tracker = PersonTracker()

        self.cropper = PersonCropper()

        self.attribute_predictor = (
            attribute_predictor
        )

        self.smoother = AttributeSmoother(
            window_size=SMOOTHING_WINDOW_SIZE,
            max_missing_frames=(
                SMOOTHING_MAX_MISSING_FRAMES
            )
        )

    def process_frame(
        self,
        frame: np.ndarray
    ) -> dict:

        detections = self.detector.detect(
            frame
        )

        tracked_objects = self.tracker.update(
            detections
        )

        annotated_frame = frame.copy()

        detected_objects = []

        active_track_ids = {
            object_info["track_id"]
            for object_info in tracked_objects
            if object_info["track_id"] >= 0
        }

        self.smoother.mark_missing(
            active_track_ids
        )

        for object_info in tracked_objects:

            track_id = object_info[
                "track_id"
            ]

            bbox = object_info[
                "bbox"
            ]

            confidence = object_info[
                "confidence"
            ]

            crop_img = self.cropper.crop(
                frame,
                bbox
            )

            if crop_img is None:
                continue

            attributes = {}
            attribute_confidence = {}

            if self.attribute_predictor is not None:

                prediction = (
                    self.attribute_predictor.predict(
                        crop_img
                    )
                )

                if prediction is not None:

                    attributes = prediction.get(
                        "attributes",
                        {}
                    )

                    attribute_confidence = (
                        prediction.get(
                            "confidence_scores",
                            {}
                        )
                    )

            smoothed_attributes = (
                self.smoother.update(
                    track_id,
                    attributes
                )
            )

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                BOX_THICKNESS
            )

            label_parts = []

            if SHOW_TRACK_ID:
                label_parts.append(
                    f"ID: {track_id}"
                )

            if SHOW_CONFIDENCE:
                label_parts.append(
                    f"{confidence:.2f}"
                )

            label = " | ".join(
                label_parts
            )

            label_y = max(
                y1 - 10,
                20
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 255, 0),
                FONT_THICKNESS,
                cv2.LINE_AA
            )

            detected_objects.append(
                {
                    "track_id": track_id,

                    "bbox": bbox,

                    "crop_img": crop_img,

                    "detection_confidence": (
                        confidence
                    ),

                    "attributes": (
                        smoothed_attributes
                    ),

                    "attribute_confidence": (
                        attribute_confidence
                    )
                }
            )

        return {
            "frame": annotated_frame,
            "detected_objects": detected_objects
        }

    def run(
        self,
        source: Union[int, str],
        show: bool = True
    ):

        reader = VideoReader(
            source
        )

        reader.open()

        previous_time = time.time()

        try:

            while True:

                ret, frame = reader.read()

                if not ret:
                    break

                result = self.process_frame(
                    frame
                )

                current_time = time.time()

                elapsed = (
                    current_time
                    - previous_time
                )

                if elapsed > 0:
                    fps = 1.0 / elapsed
                else:
                    fps = 0.0

                previous_time = current_time

                display_frame = result[
                    "frame"
                ]

                if SHOW_FPS:

                    cv2.putText(
                        display_frame,
                        f"FPS: {fps:.1f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                if SHOW_PERSON_COUNT:

                    person_count = len(
                        result[
                            "detected_objects"
                        ]
                    )

                    cv2.putText(
                        display_frame,
                        f"Persons: {person_count}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                if show:

                    cv2.imshow(
                        WINDOW_NAME,
                        display_frame
                    )

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key == ord("q"):
                        break

        finally:

            reader.release()

            cv2.destroyAllWindows()

            self.tracker.reset()

            self.smoother.reset()


if __name__ == "__main__":

    pipeline = PersonPipeline()

    pipeline.run(
        source=0
    )