
import os, json, numpy as np, pandas as pd
def _try_load_bundle(art_dir):
    import joblib
    p = os.path.join(art_dir, "complete_model_thresholded_bundle.joblib")
    if not os.path.exists(p):
        return None
    obj = joblib.load(p)
    pipe = cols = labels = thr_map = None
    if isinstance(obj, dict):
        pipe   = obj.get("pipeline") or obj.get("pipe") or obj.get("model")
        cols   = obj.get("feature_cols") or obj.get("X_cols") or obj.get("cols")
        labels = obj.get("label_names") or obj.get("classes")
        thr_map= obj.get("thresholds") or obj.get("thr_map")
    else:
        pipe = obj
        labels = getattr(pipe, "classes_", None)
    return {"pipe": pipe, "cols": cols, "labels": labels, "thr_map": thr_map}
def _load_threshold_map(artifacts_dir="artifacts"):
    path = os.path.join(artifacts_dir, "thresholds.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
def load_artifacts(artifacts_dir="artifacts"):
    import joblib
    b = _try_load_bundle(artifacts_dir)
    if b and b["pipe"] is not None:
        return b, None, None
    p = os.path.join(artifacts_dir, "pipeline_ovr_logreg.joblib")
    f = os.path.join(artifacts_dir, "feature_cols.csv")
    l = os.path.join(artifacts_dir, "label_names.csv")
    if not (os.path.exists(p) and os.path.exists(f) and os.path.exists(l)):
        return None, None, None
    return {"pipe": joblib.load(p), "cols_csv": f, "labels_csv": l, "thr_map": _load_threshold_map(artifacts_dir)}, f, l
def predict_labels(feats: dict, artifacts_dir="artifacts", threshold=0.5):
    loaded, fcols, lnames = load_artifacts(artifacts_dir)
    if loaded is None:
        return None, None
    pipe = None; cols = None; labels = None; thr_map = loaded.get("thr_map")
    if "cols_csv" in loaded:
        cols = pd.read_csv(loaded["cols_csv"], header=None)[0].tolist()
        labels = pd.read_csv(loaded["labels_csv"], header=None)[0].tolist()
        pipe = loaded["pipe"]
    else:
        pipe   = loaded["pipe"]
        cols   = loaded["cols"] or list(feats.keys())
        labels = loaded["labels"] or getattr(pipe, "classes_", None)
        if labels is None:
            raise RuntimeError("El bundle no incluye 'label_names' y el pipeline no expone classes_.")
    row = {c: np.nan for c in cols}
    for k, v in feats.items():
        if k in row: row[k] = v
    X = pd.DataFrame([row])[cols]
    y = pipe.predict_proba(X)[0]
    scores = {labels[i]: float(y[i]) for i in range(len(labels))}
    thr_map = thr_map or {}
    activos = []
    for lbl, s in scores.items():
        thr_lbl = float(thr_map.get(lbl, threshold))
        if s >= thr_lbl:
            activos.append(lbl)
    return activos, scores
