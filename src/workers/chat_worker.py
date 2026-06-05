"""Handles incoming user messages. Reads shared state, calls RAG."""
from typing import Optional
from src.workers.shared_state import SharedPostureState
from rag_query import answer


class ChatWorker:
    def __init__(self, shared: SharedPostureState):
        self.shared = shared

    def respond(self, user_message: str) -> dict:
        state = self.shared.snapshot()
        if state is None:
            return {"text": "Camera is starting up - give me a moment.",
                    "retrieved": [], "grounded": False}
        return answer(user_message, state)
