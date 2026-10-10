"""
FastAPI Web Backend Application
-------------------------------
Provides REST API endpoints for face detection, recognition, registration,
and database management.
"""

import sys
import os
import base64
import cv2
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add repository root directory to sys.path to reuse face_engine and database
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from face_engine import FaceEngine
from database import FaceDatabase
from backend.config import YUNET_PATH, SFACE_PATH, DB_PATH, API_SECRET_KEY, ALLOWED_ORIGINS, DEFAULT_THRESHOLD

# Initialize FastAPI App
app = FastAPI(
    title="Face Detection & Recognition API",
    description="FastAPI Backend for Face Detection and Recognition System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engine & Database
try:
    engine = FaceEngine(yunet_path=YUNET_PATH, sface_path=SFACE_PATH, score_threshold=0.60)
    db = FaceDatabase(db_path=DB_PATH)
    print(f"[Backend System] Engine & Database loaded successfully. Database contains {len(db.get_registered_people())} registered people.")
except Exception as e:
    print(f"[Backend System Error] Failed to initialize engine: {e}")
    engine = None
    db = None


# Helper Functions for Image Processing
def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decodes a base64 encoded image string (data URL or raw base64) into an OpenCV BGR numpy array."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")


def encode_image_to_base64(img: np.ndarray, format: str = ".jpg") -> str:
    """Encodes an OpenCV BGR numpy array to a JPEG base64 data URL string."""
    _, buffer = cv2.imencode(format, img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


# Pydantic Request Models
class RecognizeRequest(BaseModel):
    image: str
    threshold: Optional[float] = DEFAULT_THRESHOLD


class RegisterRequest(BaseModel):
    name: str
    samples: List[str]  # List of 5 base64 image strings
    consent: bool


# Security Dependency for Admin Operations
def verify_admin_key(x_api_key: Optional[str] = Header(None)):
    if API_SECRET_KEY and API_SECRET_KEY != "admin123":
        if x_api_key != API_SECRET_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header for administrative operation."
            )
    return True


# Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Face Detection & Recognition API is operational.",
        "health": "/health",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint returning system status."""
    return {
        "status": "healthy",
        "models_loaded": engine is not None and engine.detector is not None and engine.recognizer is not None,
        "database_people_count": len(db.get_registered_people()) if db else 0,
        "opencv_version": cv2.__version__
    }


@app.post("/api/detect")
async def detect_faces_endpoint(file: Optional[UploadFile] = File(None), image_b64: Optional[str] = Form(None)):
    """Detect faces in an uploaded image file or base64 string."""
    if not engine:
        raise HTTPException(status_code=500, detail="Face engine not initialized.")

    if file:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_b64:
        img = decode_base64_image(image_b64)
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'image_b64' must be provided.")

    if img is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")

    num_faces, faces = engine.detect_faces(img)

    face_details = []
    if num_faces > 0:
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            score = float(face[14])
            face_details.append({
                "bbox": [x, y, w, h],
                "score": round(score, 3)
            })

    annotated_img = engine.draw_faces(img, faces)
    annotated_b64 = encode_image_to_base64(annotated_img)

    return {
        "face_count": num_faces,
        "faces": face_details,
        "annotated_image": annotated_b64
    }


@app.post("/api/recognize")
def recognize_faces_endpoint(req: RecognizeRequest):
    """Detect and recognize faces in a base64 encoded frame or image."""
    if not engine or not db:
        raise HTTPException(status_code=500, detail="System engine or database not initialized.")

    img = decode_base64_image(req.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image input.")

    thresh = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
    num_faces, faces = engine.detect_faces(img)

    matches = []
    face_details = []

    if num_faces > 0:
        for i, face_det in enumerate(faces):
            x, y, w, h = [int(v) for v in face_det[:4]]
            detection_score = float(face_det[14])

            aligned = engine.align_crop(img, face_det)
            embedding = engine.extract_embedding(aligned)
            identity, sim_score = db.match_embedding(embedding, engine, threshold=thresh)

            matches.append((identity, sim_score))
            face_details.append({
                "bbox": [x, y, w, h],
                "name": identity,
                "similarity_score": round(sim_score, 3),
                "detection_score": round(detection_score, 3)
            })

    annotated_img = engine.draw_faces(img, faces, matches)
    annotated_b64 = encode_image_to_base64(annotated_img)

    return {
        "face_count": num_faces,
        "faces": face_details,
        "annotated_image": annotated_b64
    }


@app.post("/api/register")
def register_person_endpoint(req: RegisterRequest):
    """Register a new person with 5 base64 image samples."""
    if not engine or not db:
        raise HTTPException(status_code=500, detail="System engine or database not initialized.")

    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty.")

    if not req.consent:
        raise HTTPException(status_code=400, detail="Consent confirmation is required for registration.")

    exists, existing_name = db.is_registered(name)
    if exists:
        raise HTTPException(status_code=400, detail=f"Person '{existing_name}' is already registered.")

    if len(req.samples) < 1:
        raise HTTPException(status_code=400, detail="At least 1 face sample is required for registration.")

    extracted_embeddings = []
    for i, b64_sample in enumerate(req.samples):
        img = decode_base64_image(b64_sample)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Sample #{i+1} is invalid or unreadable.")

        num_faces, faces = engine.detect_faces(img)
        if num_faces == 0:
            raise HTTPException(status_code=400, detail=f"Sample #{i+1} failed: No face detected.")
        elif num_faces > 1:
            raise HTTPException(status_code=400, detail=f"Sample #{i+1} failed: Multiple faces detected. Keep only 1 face in frame.")

        aligned = engine.align_crop(img, faces[0])
        embedding = engine.extract_embedding(aligned)
        if embedding is None:
            raise HTTPException(status_code=400, detail=f"Sample #{i+1} failed: Feature extraction error.")

        extracted_embeddings.append(embedding)

    success, msg = db.register_person(name, extracted_embeddings)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {
        "status": "success",
        "message": msg,
        "registered_person": name,
        "sample_count": len(extracted_embeddings)
    }


@app.get("/api/people")
def list_registered_people():
    """List all registered identities and sample counts."""
    if not db:
        return []
    return db.get_registered_people()


@app.delete("/api/people/{person_name}")
def delete_person_endpoint(person_name: str, authenticated: bool = Depends(verify_admin_key)):
    """Delete a registered person from the database."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    success, msg = db.delete_person(person_name)
    if not success:
        raise HTTPException(status_code=404, detail=msg)

    return {
        "status": "success",
        "message": msg
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
