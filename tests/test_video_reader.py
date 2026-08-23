import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from pipeline.video_reader import VideoReader


def main():
    reader = VideoReader("data/raw/test.mp4")

    # Nếu muốn webcam:
    # reader = VideoReader(0)

    while True:
        ret, frame = reader.read()

        if not ret:
            break

        cv2.imshow("Video Reader", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()