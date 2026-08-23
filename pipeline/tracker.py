# pipeline/tracker.py

from typing import List, Dict, Any

import numpy as np
import supervision as sv

from .config import (
    TRACK_LOST_BUFFER,
    TRACK_ACTIVATION_THRESHOLD,
    TRACK_MATCHING_THRESHOLD,
    TRACKER_FRAME_RATE
)


class PersonTracker:
    """
    ByteTrack tracker.

    Input:
        Detection từ PersonDetector.

    Output:
        Detection + Track ID.
    """

    def __init__(
        self,
        track_activation_threshold: float = TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = TRACK_LOST_BUFFER,
        minimum_matching_threshold: float = TRACK_MATCHING_THRESHOLD,
        frame_rate: int = TRACKER_FRAME_RATE
    ):

        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=(
                track_activation_threshold
            ),

            lost_track_buffer=(
                lost_track_buffer
            ),

            minimum_matching_threshold=(
                minimum_matching_threshold
            ),

            frame_rate=frame_rate
        )

    def update(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Cập nhật ByteTrack bằng detection hiện tại.

        Args:
            detections:
                Kết quả từ PersonDetector.

        Returns:
            [
                {
                    "track_id": 1,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.91,
                    "class_id": 0
                }
            ]
        """

        # Không có detection
        if not detections:

            # Vẫn phải update tracker
            empty_detections = sv.Detections.empty()

            tracked = (
                self.byte_tracker
                .update_with_detections(
                    empty_detections
                )
            )

            return self._convert_result(
                tracked
            )

        # -----------------------------------------
        # Convert detection → supervision Detections
        # -----------------------------------------

        xyxy = np.array(
            [
                detection["bbox"]
                for detection in detections
            ],
            dtype=np.float32
        )

        confidence = np.array(
            [
                detection["confidence"]
                for detection in detections
            ],
            dtype=np.float32
        )

        class_id = np.array(
            [
                detection["class_id"]
                for detection in detections
            ],
            dtype=int
        )

        sv_detections = sv.Detections(
            xyxy=xyxy,

            confidence=confidence,

            class_id=class_id
        )

        # -----------------------------------------
        # ByteTrack
        # -----------------------------------------

        tracked_detections = (
            self.byte_tracker
            .update_with_detections(
                sv_detections
            )
        )

        return self._convert_result(
            tracked_detections
        )

    def _convert_result(
        self,
        tracked_detections
    ) -> List[Dict[str, Any]]:
        """
        Convert kết quả supervision
        thành format nội bộ của project.
        """

        results = []

        if (
            tracked_detections is None
            or len(tracked_detections) == 0
        ):
            return results

        boxes = tracked_detections.xyxy

        confidences = (
            tracked_detections.confidence
        )

        class_ids = (
            tracked_detections.class_id
        )

        tracker_ids = (
            tracked_detections.tracker_id
        )

        for i in range(
            len(tracked_detections)
        ):

            x1, y1, x2, y2 = map(
                int,
                boxes[i]
            )

            track_id = (
                int(tracker_ids[i])
                if tracker_ids is not None
                else -1
            )

            confidence = (
                float(confidences[i])
                if confidences is not None
                else 0.0
            )

            class_id = (
                int(class_ids[i])
                if class_ids is not None
                else 0
            )

            results.append(
                {
                    "track_id": track_id,

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "confidence": confidence,

                    "class_id": class_id
                }
            )

        return results

    def reset(self):
        """
        Reset tracker.

        Dùng khi:
            - chuyển sang video mới
            - chuyển webcam
            - restart pipeline
        """

        self.byte_tracker.reset()