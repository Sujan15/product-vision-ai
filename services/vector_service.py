# services/vector_service.py


import faiss
import json
import numpy as np
import os

class VectorService:
    def __init__(self, index_path, registry_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}. Run build_db.py first.")
        
        self.index = faiss.read_index(index_path)
        with open(registry_path, 'r') as f:
            self.registry = json.load(f)


    def search(self, embedding, threshold=0.65): # Higher threshold for ArcFace
        vector = embedding.reshape(1, -1).astype('float32')
        # Search top 1
        distances, indices = self.index.search(vector, 1)
        
        score = distances[0][0]
        idx = str(indices[0][0])

        if score > threshold and idx in self.registry:
            result = self.registry[idx]
            result['score'] = float(score)
            return result
                
        return {"id": "UNKNOWN", "name": "Unknown Product", "score": float(score)}
    