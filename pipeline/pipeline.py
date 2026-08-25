import time
import os
import shutil
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

        self.attribute_predictor = attribute_predictor

        self.smoother = AttributeSmoother(
            window_size=SMOOTHING_WINDOW_SIZE,
            max_missing_frames=SMOOTHING_MAX_MISSING_FRAMES
        )

    # PROCESS ONE FRAME
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
                bbox
            )

            if crop_img is None:
                continue

            attributes = {}
            attribute_confidence = {}

            if self.attribute_predictor is not None:

                prediction = self.attribute_predictor.predict(
                    crop_img
                )

                if prediction is not None:

                    attributes = prediction.get(
                        "attributes",
                        {}
                    )

                    attribute_confidence = prediction.get(
                        "confidence_scores",
                        {}
                    )

            smoothed_attributes = self.smoother.update(
                track_id,
                attributes
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
                    cv2.LINE_AA
                )

            detected_objects.append({
                "track_id": track_id,
                "bbox": bbox,
                "crop_img": crop_img,
                "detection_confidence": confidence,
                "attributes": smoothed_attributes,
                "attribute_confidence": attribute_confidence
            })

        return {
            "frame": annotated_frame,
            "detected_objects": detected_objects
        }

    # RUN VIDEO / WEBCAM
    def run(
        self,
        source: Union[int, str],
        show: bool = True,
        output_path: str | None = None
    ):

        reader = VideoReader(source)
        reader.open()

        writer = None
        video_fps = 25.0

        previous_time = time.time()
        frame_count = 0

        try:

            while True:

                ret, frame = reader.read()

                if not ret:
                    break

                result = self.process_frame(frame)

                current_time = time.time()

                elapsed = current_time - previous_time

                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                previous_time = current_time

                display_frame = result["frame"]

                # FPS
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

                # Person count
                if SHOW_PERSON_COUNT:

                    person_count = len(
                        result["detected_objects"]
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

                # Create writer
                if output_path is not None and writer is None:

                    video_height, video_width = (
                        display_frame.shape[:2]
                    )
                    # Tạo thư mục output nếu chưa có
                    output_dir = os.path.dirname(output_path)

                    if output_dir:
                        os.makedirs(
                            output_dir,
                            exist_ok=True
                        )

                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )

                    writer = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        video_fps,
                        (video_width, video_height)
                    )

                    if not writer.isOpened():

                        raise RuntimeError(
                            f"Không thể tạo video output: "
                            f"{output_path}"
                        )

                # Write frame
                if writer is not None:
                    writer.write(display_frame)

                # Show local
                if show:

                    cv2.imshow(
                        WINDOW_NAME,
                        display_frame
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
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
            print(
                f"Output video: {output_path}"
            )

    # RUN COLAB
    def run_colab(
        self,
        test_dir="/content/PAR/tests",
        result_dir="/content/PAR/tests/results"
    ):
        """
        Chạy pipeline trực tiếp trên Google Colab.
        - Chọn video có sẵn
        - Upload video mới
        - Bỏ qua video đã có kết quả
        - Q hoặc Enter để thoát
        """

        from google.colab import files

        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        extensions = (
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm"
        )

        # VIDEO no results
        videos = [
            f for f in os.listdir(test_dir)
            if f.lower().endswith(extensions)
            and not os.path.exists(
                os.path.join(
                    result_dir,
                    f"result_{os.path.splitext(f)[0]}.mp4"
                )
            )
        ]

        print("\n VIDEO CHƯA TEST ")

        if videos:

            for i, video in enumerate(videos, 1):
                print(f"{i}. {video}")

        else:

            print("Không có video chưa test.")

        print("\nU. Upload video")
        print("Q. Thoát")

        choice = input("\nChọn: ").strip()

        # Exits
        if not choice or choice.lower() == "q":

            print("Đã dừng chương trình.")
            return

        # UPLOAD
        if choice.lower() == "u":

            print("\nChọn video từ máy...")

            uploaded = files.upload()

            if not uploaded:

                print("Không có video được upload.")
                return

            name = next(iter(uploaded))

            source = os.path.join(
                test_dir,
                name
            )

            shutil.move(
                name,
                source
            )

        # CHỌN VIDEO CÓ SẴN
        elif choice.isdigit():

            index = int(choice) - 1

            if index < 0 or index >= len(videos):

                print("Lựa chọn không hợp lệ.")
                return

            name = videos[index]

            source = os.path.join(
                test_dir,
                name
            )

        else:

            print("Lựa chọn không hợp lệ.")
            return

        # OUTPUT
        base_name = os.path.splitext(name)[0]

        output = os.path.join(
            result_dir,
            f"result_{base_name}.mp4"
        )

        # Check results
        if os.path.exists(output):

            print(
                f"\n⚠️ Video này đã được xử lý!"
            )

            print(
                f"Kết quả: {output}"
            )

            return

        # RUN PIPELINE
        print("\n==============================")
        print(f"Input : {source}")
        print(f"Output: {output}")
        print("==============================")
        print("\nĐang xử lý...\n")

        self.run(
            source=source,
            show=False,
            output_path=output
        )
        print("✅ HOÀN TẤT")
        print(f"Kết quả: {output}")
