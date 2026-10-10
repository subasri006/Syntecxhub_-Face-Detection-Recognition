"""
Standalone Live Webcam Face Recognition Script
-----------------------------------------------
Streams video from webcam, detects faces, compares face embeddings
against known_faces/faces.json, and renders real-time recognition results.

Usage:
  python recognize_faces.py
  python recognize_faces.py --camera 0 --threshold 0.363
"""

import sys
import argparse
import cv2
from face_engine import FaceEngine
from database import FaceDatabase


def run_live_recognition(camera_id=0, threshold=0.363):
    """
    Runs real-time face detection & recognition on live webcam video stream.
    """
    print("=" * 60)
    print("      LIVE WEBCAM FACE DETECTION & RECOGNITION")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' or ESC to exit live recognition.")
    print(f"  - Current Cosine Threshold: {threshold}")
    print("=" * 60)

    try:
        engine = FaceEngine()
        db = FaceDatabase()
    except Exception as e:
        print(f"[Error] Engine or Database initialization failed: {e}")
        return

    registered_people = db.get_registered_people()
    print(f"[Info] Registered People in Database: {len(registered_people)}")
    for person in registered_people:
        print(f"  - {person['name']} ({person['sample_count']} samples)")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[Error] Could not open webcam camera index {camera_id}.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Error] Failed to read frame from camera.")
                break

            # Mirror webcam frame for natural viewing
            frame = cv2.flip(frame, 1)

            # Detect faces
            num_faces, faces = engine.detect_faces(frame)

            matches = []
            if num_faces > 0:
                for face_det in faces:
                    aligned = engine.align_crop(frame, face_det)
                    embedding = engine.extract_embedding(aligned)
                    identity, score = db.match_embedding(embedding, engine, threshold=threshold)
                    matches.append((identity, score))

            # Draw bounding boxes and labels
            annotated_frame = engine.draw_faces(frame, faces, matches)

            # Overlay information bar
            h, w = annotated_frame.shape[:2]
            cv2.rectangle(annotated_frame, (0, 0), (w, 40), (20, 20, 20), -1)

            status_str = f"Live Recognition | Faces Detected: {num_faces} | Threshold: {threshold:.3f}"
            cv2.putText(annotated_frame, status_str, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Live Face Recognition (Press 'q' to Quit)", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("[Info] Stopping live recognition...")
                break

    except Exception as e:
        print(f"[Error] Exception occurred during recognition loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Info] Webcam released and windows closed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run live webcam face recognition.")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--threshold", "-t", type=float, default=0.363, help="Cosine similarity threshold (default: 0.363).")

    args = parser.parse_args()
    run_live_recognition(camera_id=args.camera, threshold=args.threshold)


if __name__ == "__main__":
    main()
