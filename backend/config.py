"""
Backend Configuration Module
----------------------------
Loads environment variables and system settings for FastAPI backend.
"""

import os

# Root directory of the repository
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec_int8.onnx")
if not os.path.exists(SFACE_PATH):
    SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

# Database Path
DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "known_faces", "faces.json"))

# Security / Authentication
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "admin123")

# CORS Allowed Origins (Comma-separated list or '*')
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# Default Recognition Threshold
DEFAULT_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.363"))
