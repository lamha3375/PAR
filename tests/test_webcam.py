import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from pipeline.threaded_pipeline import ThreadedPersonPipeline


def main():
    pipeline = ThreadedPersonPipeline()

    pipeline.run_threaded(
        source=0,
        show=True,
        output_path=None,
        queue_size=2,
        drop_frames=True,
    )


if __name__ == "__main__":
    main()