
import os, glob
from typing import List, Tuple
import numpy as np
import pandas as pd

KPTS = 17

def find_all_csv(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "pose_data.csv"), recursive=True))

def load_pose_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["timestamp_ms","label","width","height"]
    for n in need:
        if n not in df.columns:
            raise ValueError(f"{path} missing column: {n}")
    return df

def normalize_xy(kxy: np.ndarray, width: int, height: int, mode: str="image") -> np.ndarray:
    if mode == "image":
        out = kxy.copy()
        out[:,0] = out[:,0] / max(1, width)
        out[:,1] = out[:,1] / max(1, height)
        return out
    elif mode == "center_hips":
        out = kxy.copy()
        cx = (out[11,0] + out[12,0]) / 2.0
        cy = (out[11,1] + out[12,1]) / 2.0
        out[:,0] = (out[:,0] - cx) / max(1, width)
        out[:,1] = (out[:,1] - cy) / max(1, height)
        return out
    else:
        raise ValueError(f"Unknown norm mode: {mode}")

def df_to_sequences(df: pd.DataFrame, win: int, stride: int, features: str, norm: str):
    """Return (X, labels_str) without numeric mapping yet."""
    rows = []
    for _,r in df.iterrows():
        kxy = np.zeros((KPTS,2), dtype=np.float32)
        kcf = np.zeros((KPTS,), dtype=np.float32)
        for i in range(KPTS):
            kxy[i,0] = r[f"kpt{i}_x"]
            kxy[i,1] = r[f"kpt{i}_y"]
            kcf[i]   = r[f"kpt{i}_conf"]
        rows.append((int(r["timestamp_ms"]), str(r["label"]), int(r["width"]), int(r["height"]), kxy, kcf))

    n = len(rows)
    feats, labels = [], []
    for start in range(0, max(0,n - win + 1), stride):
        end = start + win
        clip = rows[start:end]
        if len(clip) < win: break
        center_label = clip[len(clip)//2][1]
        seq = []
        for (_ts, _lb, w, h, kxy, kcf) in clip:
            kxy_norm = normalize_xy(kxy, w, h, mode=norm)
            if features == "xy":
                seq.append(kxy_norm.reshape(-1))
            elif features == "xyconf":
                seq.append(np.concatenate([kxy_norm.reshape(-1), kcf.astype(np.float32)]))
            else:
                raise ValueError("features must be 'xy' or 'xyconf'")
        feats.append(np.stack(seq, axis=0))
        labels.append(center_label)

    if not feats:
        return np.zeros((0,win,34),dtype=np.float32), []
    X = np.stack(feats, axis=0).astype(np.float32)
    return X, labels

def load_dataset(root: str, win: int, stride: int, features: str, norm: str):
    """Return (X, y_id, classes, sessions) with a global class mapping.
    'sessions' is a list of session IDs (one per sequence) derived from each CSV's parent dir name.
    """
    paths = find_all_csv(root)
    if not paths:
        raise RuntimeError(f"No pose_data.csv under {root}")

    all_X, all_labels, all_sessions = [], [], []

    for p in paths:
        df = load_pose_csv(p)
        X_part, labels_part = df_to_sequences(df, win, stride, features, norm)
        if len(X_part) == 0:
            continue
        sess_id = os.path.basename(os.path.dirname(p))
        all_X.append(X_part)
        all_labels.extend(labels_part)
        all_sessions.extend([sess_id]*len(X_part))

    if not all_X:
        return np.zeros((0,win,34),dtype=np.float32), np.zeros((0,),dtype=np.int64), [], []

    X = np.concatenate(all_X, axis=0)
    classes = sorted(list({str(v) for v in all_labels}))
    cls2id = {c:i for i,c in enumerate(classes)}
    y = np.array([cls2id[str(v)] for v in all_labels], dtype=np.int64)
    return X, y, classes, np.array(all_sessions)
