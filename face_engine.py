"""
Face Engine Module
------------------
Handles face detection using OpenCV YuNet (FaceDetectorYN)
and face recognition/embedding extraction using OpenCV SFace (FaceRecognizerSF).
"""

import os
import cv2
import numpy as np


class FaceEngine:
    def __init__(self, yunet_path=None, sface_path=None, score_threshold=0.6, nms_threshold=0.3, top_k=5000):
        """
        Initialize YuNet face detector and SFace face recognizer.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if yunet_path is None:
            yunet_path = os.path.join(base_dir, 'models', 'face_detection_yunet_2023mar.onnx')
        if sface_path is None:
            sface_path = os.path.join(base_dir, 'models', 'face_recognition_sface_2021dec_int8.onnx')
            if not os.path.exists(sface_path):
                # Fallback to FP32 model if int8 is missing
                sface_path = os.path.join(base_dir, 'models', 'face_recognition_sface_2021dec.onnx')

        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        self.detector = None
        self.recognizer = None

        self._load_models()

    def _load_models(self):
        """Load detector and recognizer ONNX models with fallback error handling."""
        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"YuNet face detection model not found at '{self.yunet_path}'")

        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=self.yunet_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YuNet face detector: {e}")

        if not os.path.exists(self.sface_path):
            raise FileNotFoundError(f"SFace face recognition model not found at '{self.sface_path}'")

        try:
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self.sface_path,
                config=""
            )
        except Exception as e:
            # Check fallback to non-quantized model if int8 failed
            base_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_path = os.path.join(base_dir, 'models', 'face_recognition_sface_2021dec.onnx')
            if os.path.exists(fallback_path) and fallback_path != self.sface_path:
                print(f"[FaceEngine Warning] SFace model '{self.sface_path}' failed: {e}. Retrying with fallback '{fallback_path}'...")
                self.sface_path = fallback_path
                self.recognizer = cv2.FaceRecognizerSF.create(model=self.sface_path, config="")
            else:
                raise RuntimeError(f"Failed to initialize SFace face recognizer: {e}")

    def detect_faces(self, image):
        """
        Detect faces in a BGR image.
        Returns (count, faces_array).
        faces_array is shape (N, 15) or None if no face detected.
        """
        if image is None or image.size == 0:
            return 0, None

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return 0, None

        return len(faces), faces

    def align_crop(self, image, face_det):
        """
        Align and crop face region given an image and a single face detection array (15 elements).
        Returns a (112, 112, 3) aligned BGR image.
        """
        if image is None or face_det is None:
            return None
        aligned_face = self.recognizer.alignCrop(image, face_det)
        return aligned_face

    def extract_embedding(self, aligned_face):
        """
        Extract 128-d feature embedding vector from an aligned face image.
        Returns a 1D float32 numpy array.
        """
        if aligned_face is None or aligned_face.size == 0:
            return None
        feature = self.recognizer.feature(aligned_face)
        return feature.flatten().astype(np.float32)

    def compare_embeddings(self, emb1, emb2):
        """
        Calculate Cosine Similarity between two embedding vectors.
        Returns float similarity score between -1.0 and 1.0 (1.0 = identical).
        """
        if emb1 is None or emb2 is None:
            return 0.0

        e1 = np.asarray(emb1, dtype=np.float32).reshape(1, -1)
        e2 = np.asarray(emb2, dtype=np.float32).reshape(1, -1)

        try:
            score = self.recognizer.match(e1, e2, cv2.FaceRecognizerSF_FR_COSINE)
            return float(score)
        except Exception:
            # Fallback to direct numpy cosine similarity if OpenCV match API fails
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(e1, e2.T)[0][0] / (norm1 * norm2))

    def draw_faces(self, image, faces, matches=None, draw_landmarks=True):
        """
        Draw bounding boxes, labels, and landmarks on an image.
        matches: list of (name, similarity_score) tuples corresponding to each face in faces.
        Returns annotated BGR image copy.
        """
        if image is None:
            return None

        canvas = image.copy()
        if faces is None or len(faces) == 0:
            return canvas

        for i, face in enumerate(faces):
            # Parse bounding box
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            detection_score = float(face[14])

            name = "Face"
            match_score = None
            is_known = False

            if matches and i < len(matches):
                name, match_score = matches[i]
                if name != "Unknown":
                    is_known = True

            # Color scheme: Green for recognized identity, Orange/Red for Unknown / Unmatched
            box_color = (0, 200, 80) if is_known else (0, 140, 255)
            
            # Format text label
            if match_score is not None and name != "Unknown":
                label_text = f"{name} ({match_score:.2f})"
            elif name == "Unknown" and match_score is not None:
                label_text = f"Unknown ({match_score:.2f})"
            elif name != "Face":
                label_text = f"{name}"
            else:
                label_text = f"Face ({detection_score:.2f})"

            # Draw bounding box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 2)

            # Draw label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            banner_y1 = max(0, y - text_h - 10)
            banner_y2 = y
            banner_x1 = x
            banner_x2 = max(x + w, x + text_w + 10)

            # Solid background rectangle for text readability
            cv2.rectangle(canvas, (banner_x1, banner_y1), (banner_x2, banner_y2), box_color, -1)
            cv2.putText(canvas, label_text, (banner_x1 + 5, banner_y2 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Draw 5 facial landmarks (eyes, nose, mouth corners)
            if draw_landmarks:
                landmarks = face[4:14].reshape((5, 2)).astype(int)
                lm_colors = [
                    (255, 255, 0), # Right eye (cyan)
                    (255, 255, 0), # Left eye (cyan)
                    (0, 255, 255), # Nose tip (yellow)
                    (255, 0, 255), # Right mouth corner (magenta)
                    (255, 0, 255)  # Left mouth corner (magenta)
                ]
                for (lx, ly), c in zip(landmarks, lm_colors):
                    cv2.circle(canvas, (lx, ly), 2, c, -1)

        return canvas
