# services/identity_service.py


import cv2
import numpy as np
import onnxruntime as ort

class IdentityService:
    def __init__(self, model_path, threads=30):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        # Load model
        self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, patch):
        # 1. Professional Letterboxing (Preserve Aspect Ratio)
        h, w = patch.shape[:2]
        size = 224
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(patch, (new_w, new_h))
        
        # Create black canvas (float32)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # 2. Pre-processing
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Explicitly set to float32 immediately
        img = img.astype(np.float32) / 255.0
        
        # 3. Normalization (Using explicit float32 arrays to prevent 'double' conversion)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # 4. Final Formatting
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0) # Add batch dimension
        
        # THE CRITICAL FIX: Explicitly cast to float32 before running ONNX
        img = img.astype(np.float32)

        # 5. Run Inference
        embedding = self.session.run(None, {self.input_name: img})[0]
        
        # 6. Normalize Vector (for Cosine Similarity)
        feat = embedding.flatten()
        return feat / (np.linalg.norm(feat) + 1e-10) # Added epsilon to prevent div by zero
    
    # # From ConvNeXt → ViT
    # def get_embedding(self, patch):
    #     h, w = patch.shape[:2]
    #     size = 224
    #     scale = size / max(h, w)
    #     new_w, new_h = int(w * scale), int(h * scale)

    #     resized = cv2.resize(patch, (new_w, new_h))
    #     canvas = np.zeros((size, size, 3), dtype=np.uint8)
    #     y_offset = (size - new_h) // 2
    #     x_offset = (size - new_w) // 2
    #     canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    #     img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    #     img = img.astype(np.float32) / 255.0

    #     # ViT normalization
    #     mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    #     std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    #     img = (img - mean) / std

    #     img = np.transpose(img, (2, 0, 1))
    #     img = np.expand_dims(img, axis=0).astype(np.float32)

    #     embedding = self.session.run(None, {self.input_name: img})[0]

    #     # SAFE OUTPUT HANDLING
    #     if len(embedding.shape) == 3:
    #         embedding = embedding[:, 0, :]  # CLS token

    #     feat = embedding.flatten()
    #     return feat / (np.linalg.norm(feat) + 1e-10)
