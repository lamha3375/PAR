from pathlib import Path
from typing import Generator, Union

import cv2

from .config import DEFAULT_VIDEO_FPS


class VideoReader:

    def __init__(
        self,
        source: Union[int, str, Path],
        width: int | None = None,
        height: int | None = None
    ):
        self.source = source
        self.width = width
        self.height = height
        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Không thể mở source: {self.source}"
            )

        if self.width is not None:
            self.capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.width
            )

        if self.height is not None:
            self.capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.height
            )

        return self

    def read(self):
        if self.capture is None:
            raise RuntimeError(
                "VideoReader chưa được mở. "
                "Hãy gọi open() trước."
            )

        return self.capture.read()

    def frames(self) -> Generator:
        if self.capture is None:
            self.open()

        while True:
            ret, frame = self.capture.read()

            if not ret:
                break

            yield frame

    def get_fps(self) -> float:
        if self.capture is None:
            return 0.0

        fps = self.capture.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            return DEFAULT_VIDEO_FPS

        return fps

    def get_width(self) -> int:
        if self.capture is None:
            return 0

        return int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

    def get_height(self) -> int:
        if self.capture is None:
            return 0

        return int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.release()