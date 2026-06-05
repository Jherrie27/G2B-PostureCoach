"""Thread-safe holder for the current PostureState. One per process."""
import copy
import threading
from collections import deque
from typing import Optional, List

from src.state.posture_state import PostureState


class SharedPostureState:
    def __init__(self, history_size: int = 4500):  # 5 min @ 15 fps
        self._lock = threading.Lock()
        self._current: Optional[PostureState] = None
        self._history: deque = deque(maxlen=history_size)

    def update(self, state: PostureState) -> None:
        with self._lock:
            self._current = state
            self._history.append(state)

    def snapshot(self) -> Optional[PostureState]:
        with self._lock:
            return copy.deepcopy(self._current)

    def history_last_seconds(self, seconds: float) -> List[PostureState]:
        with self._lock:
            if not self._history:
                return []
            cutoff = self._history[-1].timestamp.timestamp() - seconds
            return [s for s in self._history
                    if s.timestamp.timestamp() >= cutoff]
