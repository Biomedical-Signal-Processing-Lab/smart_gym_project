
import os, json, numpy as np, torch

def class_weights_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
