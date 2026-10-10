"""
Standalone Face Registration Script
-----------------------------------
Captures 5 face samples via webcam for consenting individuals,
extracts SFace embeddings, and registers them in the local database.

Usage:
  python register_face.py --name "John Doe"
  python register_face.py --name "John Doe" --camera 0 --samples 5
"""

import os
import sys
import argparse
import cv2
import numpy as np
from face_engine import FaceEngine
from database import FaceDatabase


def register_person_webcam(person_name, camera_id=0, target_samples=5):
    """
    Interactively captures face samples from webcam and saves to database.
    """
    name = person_name.strip()
    if not name:
        print("[Error] Person name cannot be empty.")
        return False

    db = FaceDatabase()
    exists, existing_name = db.is_registered(name)
    if exists:
        print(f"[Error] Person '{existing_name}' is already registered in the database.")
        return False

    print("=" * 60)
    print(f"       FACE REGISTRATION: '{name}'")
    print("=" * 60)
    print("Instructions:")
    print("  1. Position your face clearly in front of the webcam.")
    print("  2. Ensure good lighting and only ONE person is visible.")
    print("  3. Press SPACE or 'c' to capture a sample frame.")
    print(f"  4. Collect exactly {target_samples} clear samples.")
    print("  5. Press 'q' or ESC at any time to cancel registration.")
    print("=" * 60)

    try:
        engine = FaceEngine()
    except Exception as e:
        print(f"[Error] Engine initialization failed: {e}")
        return False

    # Open webcam
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[Error] Unable to access camera device {camera_id}.")
        return False

    samples = []
    sample_count = 0
    feedback_message = "Press SPACE or 'c' to capture sample."
    feedback_color = (255, 255, 255)

    try:
        while sample_count < target_samples:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Error] Failed to read frame from camera.")
                break

            # Mirror frame for intuitive webcam view
            frame = cv2.flip(frame, 1)
            num_faces, faces = engine.detect_faces(frame)

            display = frame.copy()

            # Status overlay banner
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w, 75), (30, 30, 30), -1)
            
            title_str = f"Registering: {name} [{sample_count}/{target_samples} samples]"
            cv2.putText(display, title_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, feedback_message, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, feedback_color, 1)

            # Draw bounding box for visual guidance
            if num_faces == 1:
                face = faces[0]
                aligned = engine.align_crop(frame, face)

                # Draw bounding box
                display = engine.draw_faces(display, faces, matches=[("Sample Candidate", None)], draw_landmarks=True)
            elif num_faces > 1:
                display = engine.draw_faces(display, faces, draw_landmarks=False)
                cv2.putText(display, "WARNING: Multiple faces detected!", (15, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(f"Register Person - {name}", display)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("[Info] Registration cancelled by user.")
                break

            if key in (32, ord('c'), ord('C')):  # SPACE or 'c' key
                if num_faces == 0:
                    feedback_message = "No face detected! Look directly at the camera."
                    feedback_color = (0, 0, 255)
                elif num_faces > 1:
                    feedback_message = "Multiple faces detected! Please keep only one face in frame."
                    feedback_color = (0, 0, 255)
                else:
                    # Single face detected! Extract embedding
                    aligned_face = engine.align_crop(frame, faces[0])
                    embedding = engine.extract_embedding(aligned_face)
                    if embedding is not None:
                        samples.append(embedding)
                        sample_count += 1
                        feedback_message = f"Sample #{sample_count} captured successfully!"
                        feedback_color = (0, 255, 0)
                        print(f"  [+] Captured sample {sample_count}/{target_samples}")
                    else:
                        feedback_message = "Failed to extract features from face sample. Try again."
                        feedback_color = (0, 0, 255)

        # Release camera window
        cap.release()
        cv2.destroyAllWindows()

        if sample_count == target_samples:
            success, msg = db.register_person(name, samples)
            if success:
                print("\n" + "=" * 60)
                print(f"[SUCCESS] {msg}")
                print("=" * 60)
                return True
            else:
                print(f"[Error] Failed to save registration: {msg}")
                return False
        else:
            print(f"[Info] Registration incomplete ({sample_count}/{target_samples} samples collected). Nothing saved.")
            return False

    except Exception as e:
        print(f"[Error] An error occurred during registration: {e}")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        return False


def main():
    parser = argparse.ArgumentParser(description="Register a new person's face via webcam.")
    parser.add_argument("--name", "-n", required=True, help="Full name of person to register.")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--samples", "-s", type=int, default=5, help="Number of face samples to collect (default: 5).")

    args = parser.parse_args()
    register_person_webcam(args.name, camera_id=args.camera, target_samples=args.samples)


if __name__ == "__main__":
    main()
