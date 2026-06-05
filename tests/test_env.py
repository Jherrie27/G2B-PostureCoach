"""Smoke test: every critical import must succeed and emit no protobuf warning."""
import warnings
warnings.filterwarnings("error", message=".*protobuf.*")  # promote to error

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import lightgbm as lgb
import groq
import streamlit
from rank_bm25 import BM25Okapi

# NOTE: chromadb + sentence_transformers (RAG stack) are verified by
# tests/test_env_rag.py at Phase 6 — they need a C++ toolchain on Windows
# and are not required for Phases 1-5.

print("mediapipe:", mp.__version__)
print("cv2:      ", cv2.__version__)
print("numpy:    ", np.__version__)
print("lightgbm: ", lgb.__version__)

# Smoke-test MediaPipe Pose (this is where protobuf usually explodes)
pose = mp.solutions.pose.Pose(static_image_mode=True)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
result = pose.process(dummy)
print("MediaPipe Pose loaded and ran on dummy frame: OK")
print("ALL IMPORTS OK")
