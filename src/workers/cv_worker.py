"""Background thread: capture webcam -> run pipeline -> update SharedPostureState."""
import threading
import time

from src.cv.pipeline import PosturePipeline
from src.workers.shared_state import SharedPostureState
from src.utils.camera import open_working_camera


class CVWorker(threading.Thread):
    def __init__(self,
                 shared: SharedPostureState,
                 camera_index: int = 0,
                 width: int = 640,
                 height: int = 480,
                 model_complexity: int = 1,
                 target_fps: int = 15):
        super().__init__(daemon=True)
        self.shared = shared
        self.camera_index = camera_index
        self.width, self.height = width, height
        self.model_complexity = model_complexity
        self.frame_interval = 1.0 / target_fps
        self.stop_event = threading.Event()
        # Latest BGR frame for UI overlay
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def run(self):
        preferred = self.camera_index if self.camera_index != 0 else None
        cap, _ = open_working_camera(width=self.width, height=self.height,
                                     preferred=preferred)
        pipe = PosturePipeline(model_complexity=self.model_complexity)

        try:
            while not self.stop_event.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                state = pipe.step(frame)
                if state is not None:
                    self.shared.update(state)
                with self._frame_lock:
                    self._latest_frame = frame
                elapsed = time.time() - t0
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
        finally:
            cap.release()
            pipe.close()

    def stop(self):
        self.stop_event.set()

    def latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()
