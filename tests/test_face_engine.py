"""
Unit Tests for Face Engine Module (face_engine.py)
"""

import unittest
import numpy as np
import cv2
from face_engine import FaceEngine


class TestFaceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FaceEngine()

    def test_engine_initialization(self):
        self.assertIsNotNone(self.engine.detector)
        self.assertIsNotNone(self.engine.recognizer)

    def test_detect_faces_empty_image(self):
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        count, faces = self.engine.detect_faces(empty_img)
        self.assertEqual(count, 0)
        self.assertIsNone(faces)

    def test_align_crop_and_embedding_shape(self):
        # Create dummy detection vector [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
        dummy_face = np.array([40, 40, 80, 80, 60, 60, 100, 60, 80, 80, 65, 100, 95, 100, 0.95], dtype=np.float32)
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

        aligned = self.engine.align_crop(dummy_img, dummy_face)
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.shape, (112, 112, 3))

        embedding = self.engine.extract_embedding(aligned)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (128,))

    def test_compare_embeddings_identical(self):
        emb = np.random.rand(128).astype(np.float32)
        score = self.engine.compare_embeddings(emb, emb)
        self.assertAlmostEqual(score, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
