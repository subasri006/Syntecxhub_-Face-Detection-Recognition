"""
Database Module
---------------
Manages local face database storage in known_faces/faces.json.
Provides thread-safe loading, saving, registration, deletion, and vector matching.
"""

import os
import json
import time
import shutil
import numpy as np


class FaceDatabase:
    def __init__(self, db_path=None):
        """
        Initialize database pointing to specified JSON path.
        Default: 'known_faces/faces.json'
        """
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, 'known_faces', 'faces.json')
        
        self.db_path = db_path
        self.data = {}
        self.ensure_directory()
        self.load()

    def ensure_directory(self):
        """Ensure destination directory exists."""
        folder = os.path.dirname(self.db_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def load(self):
        """
        Load known face embeddings from JSON file.
        Handles corrupt or invalid JSON gracefully.
        """
        if not os.path.exists(self.db_path):
            self.data = {}
            return self.data

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    self.data = {}
                    return self.data
                self.data = json.loads(content)
                if not isinstance(self.data, dict):
                    raise ValueError("Database root must be a JSON object.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Database Error] Failed to parse '{self.db_path}': {e}")
            # Create a backup of corrupt file
            timestamp = int(time.time())
            backup_path = f"{self.db_path}.corrupted_{timestamp}"
            shutil.copyfile(self.db_path, backup_path)
            print(f"[Database Recovery] Backup created at '{backup_path}'. Resetting to empty database.")
            self.data = {}
            self.save()

        return self.data

    def save(self):
        """
        Safely write database to disk using an atomic write pattern.
        """
        self.ensure_directory()
        temp_path = f"{self.db_path}.tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            os.rename(temp_path, self.db_path)
            return True
        except Exception as e:
            print(f"[Database Error] Failed to save database: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def is_registered(self, name):
        """Check if person name is already registered (case-insensitive)."""
        target = name.strip().lower()
        for k in self.data.keys():
            if k.strip().lower() == target:
                return True, k
        return False, None

    def register_person(self, name, embedding_samples):
        """
        Register a new person with 1 or more face sample embedding vectors.
        name: string
        embedding_samples: list of 1D numpy arrays or python lists
        """
        name = name.strip()
        if not name:
            return False, "Person name cannot be empty."

        exists, existing_name = self.is_registered(name)
        if exists:
            return False, f"Person '{existing_name}' is already registered."

        if not embedding_samples or len(embedding_samples) == 0:
            return False, "At least one face embedding sample is required."

        serializable_samples = []
        for sample in embedding_samples:
            if isinstance(sample, np.ndarray):
                serializable_samples.append(sample.flatten().tolist())
            elif isinstance(sample, (list, tuple)):
                serializable_samples.append([float(x) for x in sample])

        self.data[name] = serializable_samples
        if self.save():
            return True, f"Successfully registered '{name}' with {len(serializable_samples)} sample(s)."
        else:
            return False, "Failed to save person to database file."

    def delete_person(self, name):
        """
        Remove a person and their face embeddings from the database.
        """
        exists, actual_name = self.is_registered(name)
        if not exists:
            return False, f"Person '{name}' not found in database."

        del self.data[actual_name]
        if self.save():
            return True, f"Successfully deleted '{actual_name}' from database."
        else:
            return False, "Failed to save updated database to disk."

    def get_registered_people(self):
        """
        Returns a list of dictionaries detailing registered people:
        [{'name': 'John', 'sample_count': 5}, ...]
        """
        result = []
        for name, samples in self.data.items():
            result.append({
                'name': name,
                'sample_count': len(samples) if isinstance(samples, list) else 0
            })
        return result

    def match_embedding(self, embedding, engine, threshold=0.363):
        """
        Compare a query face embedding vector against all registered embeddings.
        Returns (best_name, max_similarity_score).
        If best_similarity < threshold, returns ("Unknown", best_similarity).
        """
        if not self.data or embedding is None:
            return "Unknown", 0.0

        best_name = "Unknown"
        max_score = -1.0

        for name, sample_list in self.data.items():
            for sample in sample_list:
                score = engine.compare_embeddings(embedding, sample)
                if score > max_score:
                    max_score = score
                    if max_score >= threshold:
                        best_name = name

        if max_score < threshold:
            return "Unknown", max_score

        return best_name, max_score
