import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    content = f.read()

catalog = json.JSONDecoder(strict=False).decode(content)

def make_text(item):
    """
    Converts a catalog item into a single descriptive string.

    We combine the name, description, category keys, and job levels
    into one sentence. This is what gets turned into a vector by the
    sentence transformer model. The richer the text, the better the
    search results.
    """
    
    keys = ", ".join(item.get("keys", []))
    levels = ", ".join(item.get("job_levels", []))
    return f"{item['name']}. {item.get('description', '')} Categories: {keys}, Job levels: {levels}."

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [make_text(item) for item in catalog]
print(f"Encoding {len(texts)} assessments...")

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

embeddings = embeddings.astype(np.float32)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "catalog.index")

with open("catalog.pkl", "wb") as f:
    pickle.dump(catalog, f)

print("Done. File saved: catalog.index, catalog.pkl")