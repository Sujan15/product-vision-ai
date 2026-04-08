# services/detection_service.py

import cv2
import numpy as np
import onnxruntime as ort

class DetectionService:
    def __init__(self, model_path, threads=30):
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opt, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image):
        img_h, img_w = image.shape[:2]
        input_size = 640
        
        # Pre-process
        img = cv2.resize(image, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        # Run Inference
        outputs = self.session.run(None, {self.input_name: img})[0]
        
        # YOLOv8 output is [1, classes+4, 8400] -> [8400, 84]
        predictions = np.squeeze(outputs).T
        
        boxes = []
        scores = []
        
        for row in predictions:
            # For SKU-110k (usually 1 class), indices 0-3 are box, 4 is score
            score = row[4] 
            if score > 0.35:
                x, y, w, h = row[:4]
                # Rescale to original image
                x1 = int((x - w/2) * img_w / input_size)
                y1 = int((y - h/2) * img_h / input_size)
                x2 = int((x + w/2) * img_w / input_size)
                y2 = int((y + h/2) * img_h / input_size)
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))

        # NMS to remove duplicates
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.45)
        
        final_results = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_results.append(boxes[i] + [scores[i]])
                
        return final_results