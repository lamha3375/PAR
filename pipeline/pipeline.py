import os
import time
from typing import Union, List, Tuple, Generator, Dict, Any
import cv2
import numpy as np

from .attribute_interface import AttributePredictorInterface
from .config import (
    BOX_THICKNESS,
    DETECTION_CONFIDENCE,
    DISPLAY_MAX_HEIGHT,
    DISPLAY_MAX_WIDTH,
    FONT_SCALE,
    FONT_THICKNESS,
    SHOW_CONFIDENCE,
    SHOW_FPS,
    SHOW_PERSON_COUNT,
    SHOW_TRACK_ID,
    SMOOTHING_MAX_MISSING_FRAMES,
    SMOOTHING_WINDOW_SIZE,
    WINDOW_NAME,
    YOLO_MODEL_PATH,
)
from .cropper import PersonCropper
from .detector import PersonDetector
from .tracker_smoother import PersonTracker, AttributeSmoother
from .video_reader import VideoReader


def _fit_frame_for_display(
    frame: np.ndarray,
    max_width: int = DISPLAY_MAX_WIDTH,
    max_height: int = DISPLAY_MAX_HEIGHT,
) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame
    scale = min(
        max_width / w,
        max_height / h,
        1.0,
    )

    if scale == 1.0:
        return frame
    new_width = max(1, int(round(w * scale)))
    new_height = max(1, int(round(h * scale)))
    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _create_display_window():
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )


class PersonPipeline:
    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        confidence: float = DETECTION_CONFIDENCE,
        device: str | None = None,
        attribute_predictor: AttributePredictorInterface | None = None,
    ):
        self.detector = PersonDetector(
            model_path=model_path,
            confidence=confidence,
            device=device,
        )

        self.tracker = PersonTracker()
        self.cropper = PersonCropper()
        self.attribute_predictor = attribute_predictor
        self.smoother = AttributeSmoother(
            window_size=SMOOTHING_WINDOW_SIZE,
            max_missing_frames=SMOOTHING_MAX_MISSING_FRAMES,
        )

    def process_frame(self, frame: np.ndarray) -> dict:
        detections = self.detector.detect(frame)
        tracked_objects = self.tracker.update(detections)
        annotated_frame = frame.copy()
        detected_objects = []
        
        active_track_ids = {
            obj["track_id"]
            for obj in tracked_objects
            if obj["track_id"] >= 0
        }

        self.smoother.mark_missing(active_track_ids)

        for object_info in tracked_objects:
            track_id = object_info["track_id"]
            bbox = object_info["bbox"]
            confidence = object_info["confidence"]

            crop_img = self.cropper.crop(
                frame,
                bbox,
            )

            if crop_img is None:
                continue

            attributes = {}
            attribute_confidence = {}

            if self.attribute_predictor is not None:
                prediction = self.attribute_predictor.predict(crop_img)

                if prediction is not None:
                    attributes = prediction.get(
                        "attributes",
                        {},
                    )

                    attribute_confidence = prediction.get(
                        "confidence_scores",
                        {},
                    )

            smoothed_attributes = self.smoother.update(
                track_id,
                attributes,
            )

            x1, y1, x2, y2 = bbox

            # Vẽ Bounding Box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                BOX_THICKNESS,
            )

            # Vẽ Label ID / Confidence
            label_parts = []

            if SHOW_TRACK_ID:
                label_parts.append(f"ID: {track_id}")

            if SHOW_CONFIDENCE:
                label_parts.append(f"{confidence:.2f}")

            label = " | ".join(label_parts)

            if label:
                label_y = max(y1 - 10, 20)

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    (0, 255, 0),
                    FONT_THICKNESS,
                    cv2.LINE_AA,
                )

            detected_objects.append(
                {
                    "track_id": track_id,
                    "bbox": bbox,
                    "crop_img": crop_img,
                    "detection_confidence": confidence,
                    "attributes": smoothed_attributes,
                    "attribute_confidence": attribute_confidence,
                }
            )

        return {
            "frame": annotated_frame,
            "detected_objects": detected_objects,
        }

    def run(
        self,
        source: Union[int, str],
        show: bool = True,
        output_path: str | None = None,
    ):
        reader = VideoReader(source)
        reader.open()

        writer = None
        video_fps = reader.get_fps()

        previous_time = time.time()
        frame_count = 0

        if show:
            _create_display_window()

        try:
            while True:
                ret, frame = reader.read()

                if not ret:
                    break

                input_height, input_width = frame.shape[:2]

                result = self.process_frame(frame)

                current_time = time.time()
                elapsed = current_time - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_time = current_time

                output_frame = result["frame"]

                # Hiển thị FPS
                if SHOW_FPS:
                    cv2.putText(
                        output_frame,
                        f"FPS: {fps:.1f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if SHOW_PERSON_COUNT:
                    person_count = len(result["detected_objects"])

                    cv2.putText(
                        output_frame,
                        f"Persons: {person_count}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if output_path is not None and writer is None:
                    output_dir = os.path.dirname(output_path)

                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    writer = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        video_fps if video_fps > 0 else 30.0,
                        (input_width, input_height),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Không thể tạo video output: {output_path}"
                        )

                if writer is not None:
                    writer.write(output_frame)

                if show:
                    preview_frame = _fit_frame_for_display(output_frame)

                    cv2.imshow(
                        WINDOW_NAME,
                        preview_frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if (
                        key == ord("q")
                        or cv2.getWindowProperty(
                            WINDOW_NAME,
                            cv2.WND_PROP_VISIBLE,
                        )
                        < 1
                    ):
                        break

                frame_count += 1

        finally:
            reader.release()

            if writer is not None:
                writer.release()

            if show:
                cv2.destroyAllWindows()

            self.tracker.reset()
            self.smoother.reset()

        print(
            f"Pipeline finished. "
            f"Processed {frame_count} frames."
        )

        if output_path is not None:
            print(f"Output video: {output_path}")

def process_video_source(
    source: Union[int, str],
    attribute_predictor: AttributePredictorInterface | None = None,
    model_path: str = YOLO_MODEL_PATH,
    confidence: float = DETECTION_CONFIDENCE,
    device: str | None = None
) -> Generator[Tuple[np.ndarray, List[int], List[np.ndarray]], None, None]:
    pipeline = PersonPipeline(
        model_path=model_path,
        confidence=confidence,
        device=device,
        attribute_predictor=attribute_predictor
    )

    reader = VideoReader(source)
    reader.open()

    try:
        while True:
            ret, frame = reader.read()
            if not ret:
                break

            result = pipeline.process_frame(frame)
            annotated_frame = result["frame"]
            track_ids = [obj["track_id"] for obj in result["detected_objects"]]
            cropped_images = [obj["crop_img"] for obj in result["detected_objects"]]

            yield annotated_frame, track_ids, cropped_images

    finally:
        reader.release()
        pipeline.tracker.reset()
        pipeline.smoother.reset()