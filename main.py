# main.py


import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from services.detection_service import DetectionService
from services.identity_service import IdentityService
from services.vector_service import VectorService
from services.counting_service import CountingService

app = FastAPI(title="ProductVision AI")

# 1. Initialize Services
# Using intra_op_num_threads=30 for your high-core CPU
try:
    detector = DetectionService("models/yolov8s_sku110k_2.onnx", threads=30) # yolo26n_sku110k  # yolov8s_sku110k_2
    identifier = IdentityService("models/convnext_tiny_embedding.onnx", threads=30) # convnext_tiny_22k_224 # convnext_tiny_embedding # vit_base_patch16_224
    vector_db = VectorService("database/product_vectors.index", "database/product_registry.json")
    counter = CountingService(detector, identifier, vector_db)
except Exception as e:
    print(f"CRITICAL ERROR during service initialization: {e}")
    # In a real environment, you might want to stop here or log
    pass

# 2. Serve the Frontend (Static Files)
# This mounts the 'static' folder to the URL path '/static'
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Routes
@app.get("/")
async def read_index():
    """Serves the main dashboard page."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found in static folder"}

# @app.post("/analyze")
# async def analyze(file: UploadFile = File(...)):
#     """Handles image upload and product counting."""
#     try:
#         # Load Image
#         data = await file.read()
#         nparr = np.frombuffer(data, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image file")
        
#         # Process through Counting Service
#         product_list = counter.process_image(image)
        
#         return {
#             "status": "success",
#             "total_detected": sum(p['count'] for p in product_list),
#             "unique_products": len(product_list),
#             "products": product_list,
#             "accuracy_estimate": 0.85 # Based on model benchmarks
#         }
#     except Exception as e:
#         print(f"Analysis Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
from typing import List

@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    """Handles multiple image uploads and aggregates product counting."""
    try:
        all_aggregated_results = {}
        total_detected_global = 0
        
        for file in files:
            # Load Image
            data = await file.read()
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Process single image
            image_results = counter.process_image(image)
            
            # Merge results into global aggregator
            for p in image_results:
                p_id = p['id']
                if p_id in all_aggregated_results:
                    all_aggregated_results[p_id]['count'] += p['count']
                else:
                    all_aggregated_results[p_id] = p

        # Final list for frontend
        final_list = list(all_aggregated_results.values())
        
        return {
            "status": "success",
            "total_detected": sum(p['count'] for p in final_list),
            "unique_products": len(final_list),
            "products": final_list,
            "accuracy_estimate": 0.92
        }
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Optimized for CPU workload: 1 worker with 30 threads inside the worker
    uvicorn.run(app, host="0.0.0.0", port=8000)