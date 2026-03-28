# build_db.py

import os
import cv2
import json
import faiss
import numpy as np
from services.identity_service import IdentityService

# =============================
# CONFIG
# =============================
GALLERY_DIR = "database/gallery"
MODEL_PATH = "models/convnext_tiny_embedding.onnx" #convnext_tiny_22k_224  #convnext_tiny_embedding #vit_base_patch16_224
INDEX_PATH = "database/product_vectors.index"
REGISTRY_PATH = "database/product_registry.json"

# =============================
# BUILD DATABASE
# =============================
def build_database():
    identifier = IdentityService(MODEL_PATH, threads=8)

    sku_embeddings = []
    registry = {}
    idx = 0

    print(f"Scanning gallery: {GALLERY_DIR}")

    for sku_name in sorted(os.listdir(GALLERY_DIR)):
        sku_path = os.path.join(GALLERY_DIR, sku_name)
        if not os.path.isdir(sku_path):
            continue

        vectors = []
        print(f"Processing SKU: {sku_name}")

        for img_file in os.listdir(sku_path):
            img_path = os.path.join(sku_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            emb = identifier.get_embedding(image)
            vectors.append(emb)

        if not vectors:
            continue

        # ✔ PROFESSIONAL: MEAN EMBEDDING PER SKU
        sku_vector = np.mean(vectors, axis=0)
        sku_vector = sku_vector / np.linalg.norm(sku_vector)

        sku_embeddings.append(sku_vector)

        registry[idx] = {
            "id": sku_name.upper().replace(" ", "_"),
            "name": sku_name.replace("_", " ")
        }
        idx += 1

    if not sku_embeddings:
        raise RuntimeError("No valid SKU embeddings found")

    # =============================
    # FAISS INDEX
    # =============================
    dim = len(sku_embeddings[0])
    index = faiss.IndexFlatIP(dim)

    embeddings_array = np.array(sku_embeddings).astype("float32")
    faiss.normalize_L2(embeddings_array)

    index.add(embeddings_array)

    faiss.write_index(index, INDEX_PATH)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"✔ Registered {len(registry)} SKUs")
    print(f"✔ FAISS index saved → {INDEX_PATH}")

if __name__ == "__main__":
    build_database()
