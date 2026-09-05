import os
import threading
import time
from queue import Queue, Empty
import cv2
from .pipeline import PersonPipeline
from .video_reader import ThreadedVideoReader
from .config import (
    WINDOW_NAME,
    SHOW_FPS,
    SHOW_PERSON_COUNT,
    SHOW_TRACK_ID,
    SHOW_CONFIDENCE,
    BOX_THICKNESS,
    FONT_SCALE,
    FONT_THICKNESS,
)
def _fit_frame_to_window(frame, window_width, window_height):
    if frame is None:
        return None
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return frame
    if window_width <= 0 or window_height <= 0:
        return frame
    scale = min(
        window_width / frame_width,
        window_height / frame_height,
    )
    new_width = max(
        1,
        int(round(frame_width * scale))
    )
    new_height = max(
        1,
        int(round(frame_height * scale))
    )
    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        ),
    )
    canvas = __import__("numpy").zeros(
        (
            window_height,
            window_width,
            3,
        ),
        dtype=frame.dtype,
    )
    x = (window_width - new_width) // 2
    y = (window_height - new_height) // 2
    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized
    return canvas

def _create_display_window():
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )

class ThreadedPersonPipeline(PersonPipeline):
    def run_threaded(
        self,
        source,
        show=True,
        output_path=None,
        queue_size=2,
        drop_frames=True,
    ):
        reader = ThreadedVideoReader(
            source=source,
            queue_size=queue_size,
            drop_frames=drop_frames,
        )
        result_queue = Queue(
            maxsize=queue_size
        )
        stop_event = threading.Event()
        writer = None
        frame_count = 0
        processing_count = 0
        previous_time = time.time()
        fps = 0.0
        worker_error = []

        def processing_worker():
            nonlocal processing_count
            try:
                while not stop_event.is_set():
                    try:
                        frame = reader.frame_queue.get(
                            timeout=0.1
                        )
                    except Empty:
                        if reader.is_finished():
                            break
                        continue
                    result = self.process_frame(frame)
                    try:
                        result_queue.put_nowait(
                            result
                        )
                    except:
                        try:
                            result_queue.get_nowait()
                        except Empty:
                            pass
                        try:
                            result_queue.put_nowait(
                                result
                            )
                        except:
                            pass
                    processing_count += 1
            except Exception as exc:
                worker_error.append(exc)
                stop_event.set()
        try:
            reader.start()
            if show:
                _create_display_window()
                cv2.resizeWindow(
                    WINDOW_NAME,
                    960,
                    720,
                )
            video_fps = reader.get_fps()
            if video_fps <= 0:
                video_fps = 30.0
            processing_thread = threading.Thread(
                target=processing_worker,
                name="ProcessingThread",
                daemon=True,
            )
            processing_thread.start()
            while not stop_event.is_set():
                try:
                    result = result_queue.get(
                        timeout=0.1
                    )
                except Empty:
                    if (
                        reader.is_finished()
                        and not processing_thread.is_alive()
                    ):
                        break
                    continue
                display_frame = result["frame"]
                frame_height, frame_width = (
                    display_frame.shape[:2]
                )
                current_time = time.time()
                elapsed = (
                    current_time
                    - previous_time
                )
                if elapsed > 0:
                    fps = 1.0 / elapsed
                previous_time = current_time
                if SHOW_FPS:
                    cv2.putText(
                        display_frame,
                        f"FPS: {fps:.1f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
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
                        cv2.LINE_AA,
                    )
                    
                if output_path is not None:
                    if writer is None:
                        output_dir = os.path.dirname(
                            output_path
                        )
                        if output_dir:
                            os.makedirs(
                                output_dir,
                                exist_ok=True,
                            )
                        fourcc = (
                            cv2.VideoWriter_fourcc(
                                *"mp4v"
                            )
                        )
                        writer = cv2.VideoWriter(
                            output_path,
                            fourcc,
                            video_fps,
                            (
                                frame_width,
                                frame_height,
                            ),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                "Không thể tạo video "
                                f"output: {output_path}"
                            )
                        print(
                            f"[OUTPUT] "
                            f"{frame_width} x "
                            f"{frame_height}"
                        )
                    writer.write(
                        display_frame
                    )
                if show:
                    try:
                        _, _, window_width, window_height = (
                            cv2.getWindowImageRect(
                                WINDOW_NAME
                            )
                        )
                    except Exception:
                        window_width = frame_width
                        window_height = frame_height
                    if (
                        window_width <= 0
                        or window_height <= 0
                    ):
                        window_width = frame_width
                        window_height = frame_height
                    preview_frame = (
                        _fit_frame_to_window(
                            display_frame,
                            window_width,
                            window_height,
                        )
                    )
                    cv2.imshow(
                        WINDOW_NAME,
                        preview_frame,
                    )
                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )
                    if key == ord("q"):
                        stop_event.set()
                        break
                    try:
                        visible = (
                            cv2.getWindowProperty(
                                WINDOW_NAME,
                                cv2.WND_PROP_VISIBLE,
                            )
                        )
                        if visible < 1:
                            stop_event.set()
                            break
                    except Exception:
                        pass
                frame_count += 1
            processing_thread.join(
                timeout=2.0
            )
            if worker_error:
                raise worker_error[0]
            if getattr(
                reader,
                "error",
                None,
            ) is not None:
                raise reader.error
        finally:
            stop_event.set()
            if hasattr(
                reader,
                "stopped",
            ):
                reader.stopped = True
            reader.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            self.tracker.reset()
            self.smoother.reset()
        print(
            "Threaded pipeline finished. "
            f"Displayed/processed results: "
            f"{frame_count}, "
            f"inference frames: "
            f"{processing_count}"
        )
        if output_path is not None:
            print(
                f"Output video: "
                f"{output_path}"
            )
