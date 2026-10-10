"""
Unit Tests for Database Module (database.py)
"""

import os
import tempfile
import unittest
import numpy as np
from database import FaceDatabase


class TestFaceDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for testing database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_faces.json")
        self.db = FaceDatabase(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_database_on_init(self):
        self.assertEqual(len(self.db.get_registered_people()), 0)

    def test_register_person_success(self):
        mock_embedding = np.random.rand(128).astype(np.float32)
        success, msg = self.db.register_person("Alice Smith", [mock_embedding])
        
        self.assertTrue(success)
        self.assertIn("Successfully registered", msg)
        
        people = self.db.get_registered_people()
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["name"], "Alice Smith")
        self.assertEqual(people[0]["sample_count"], 1)

    def test_prevent_empty_name(self):
        mock_emb = np.random.rand(128).astype(np.float32)
        success, msg = self.db.register_person("   ", [mock_emb])
        self.assertFalse(success)
        self.assertIn("cannot be empty", msg)

    def test_prevent_duplicate_registration(self):
        mock_emb = np.random.rand(128).astype(np.float32)
        self.db.register_person("Bob Johnson", [mock_emb])
        
        # Try registering same name case-insensitively
        success, msg = self.db.register_person("bob johnson", [mock_emb])
        self.assertFalse(success)
        self.assertIn("already registered", msg)

    def test_delete_person(self):
        mock_emb = np.random.rand(128).astype(np.float32)
        self.db.register_person("Charlie Brown", [mock_emb])
        self.assertEqual(len(self.db.get_registered_people()), 1)

        success, msg = self.db.delete_person("Charlie Brown")
        self.assertTrue(success)
        self.assertEqual(len(self.db.get_registered_people()), 0)

    def test_corrupt_json_recovery(self):
        # Write corrupted JSON content directly to test_faces.json
        with open(self.db_path, 'w', encoding='utf-8') as f:
            f.write("{ INVALID CORRUPTED JSON LOGIC }")

        # Re-initialize database, should recover cleanly
        corrupt_db = FaceDatabase(db_path=self.db_path)
        self.assertEqual(len(corrupt_db.get_registered_people()), 0)


if __name__ == "__main__":
    unittest.main()
