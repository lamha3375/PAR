import sys
import os
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from pipeline.threaded_pipeline import ThreadedPersonPipeline


pipeline = ThreadedPersonPipeline()

pipeline.run_threaded(
    source="tests/Test5.mp4",
    show=True,
    output_path="tests/results/threaded_result.mp4",
    queue_size=2,
    drop_frames=False
)