import os
import numpy as np


def save_embedding(path: str, emb: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, emb)


def load_embedding(path: str):
    if not os.path.exists(path):
        return None
    # Some stored embeddings may be object arrays (lists/dicts). Allow pickle when loading
    data = np.load(path, allow_pickle=True)

    # If data is a numpy array of objects (common when saving dicts/lists), normalize
    if isinstance(data, np.ndarray):
        if data.dtype == object and data.size >= 1:
            first = data.reshape(-1)[0]
            # case: saved dict with 'embedding' key (DeepFace.represent style)
            if isinstance(first, dict) and 'embedding' in first:
                return np.asarray(first['embedding'], dtype=float)
            # case: saved list/array inside object wrapper
            try:
                arr = np.asarray(first, dtype=float).reshape(-1)
                return arr
            except Exception:
                # fallback: try to convert whole array
                try:
                    return np.asarray(data.tolist(), dtype=float).reshape(-1)
                except Exception:
                    return data
        else:
            return np.asarray(data).reshape(-1)

    # Non-array objects: try to convert to numpy array
    try:
        return np.asarray(data).reshape(-1)
    except Exception:
        return data
