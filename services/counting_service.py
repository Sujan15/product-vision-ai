# services/counting_service.py
    
import os
import cv2
import time
import numpy as np

class CountingService:
    def __init__(self, detector, identifier, vector_db):
        self.detector = detector
        self.identifier = identifier
        self.vector_db = vector_db
        # Define the debug directory
        self.debug_base = r"D:\ETL DEMO WORK\New folder\product-vision\database\input"
        self._prepare_dirs()

    def _prepare_dirs(self):
        """Create initial structure for debugging."""
        for sub in ["originals", "detections", "crops"]:
            path = os.path.join(self.debug_base, sub)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def process_image(self, image):
        # 1. Generate a unique session ID based on timestamp
        session_id = str(int(time.time()))
        
        # 2. Save Original Image
        orig_path = os.path.join(self.debug_base, "originals", f"orig_{session_id}.jpg")
        cv2.imwrite(orig_path, image)

        # 3. Create a specific folder for THIS images crops
        current_crops_dir = os.path.join(self.debug_base, "crops", session_id)
        if not os.path.exists(current_crops_dir):
            os.makedirs(current_crops_dir, exist_ok=True)

        # 4. Detect Products
        boxes = self.detector.predict(image)
        
        # Create a copy for drawing detections
        detection_overlay = image.copy()
        results = {}
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf = box
            # Clip to image size
            x1, y1 = max(0, x1), max(0, y1)
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0: 
                continue

            # 5. Get Identity
            embedding = self.identifier.get_embedding(crop)
            product = self.vector_db.search(embedding)
            
            p_id = product['id']
            p_name = product['name']
            score = product.get('score', 0.0)

            # 6. Save Individual Crop INSIDE the session folder
            # Naming: boxIndex_SKUID_SimilarityScore.jpg
            crop_filename = f"box{i}_{p_id}_{score:.2f}.jpg"
            crop_path = os.path.join(current_crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            # 7. Draw on detection overlay
            color = (0, 255, 0) if p_id != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(detection_overlay, (x1, y1), (x2, y2), color, 3)
            label = f"{p_name} ({score:.2f})"
            cv2.putText(detection_overlay, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Aggregate Results
            if p_id in results:
                results[p_id]['count'] += 1
            else:
                results[p_id] = {
                    "id": p_id,
                    "name": p_name,
                    "count": 1,
                    "confidence": float(conf),
                    "match_score": float(score)
                }

        # 8. Save Visualized Detections
        det_path = os.path.join(self.debug_base, "detections", f"det_{session_id}.jpg")
        cv2.imwrite(det_path, detection_overlay)

        return list(results.values())    