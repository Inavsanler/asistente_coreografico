# src/model.py
import os, json
import numpy as np
import pandas as pd

def _try_load_bundle(art_dir: str):
    """
    Carga 'complete_model_thresholded_bundle.joblib' en formatos comunes:
    - dict: {pipeline/pipe/model, feature_cols/X_cols/cols, label_names/classes, thresholds/thr_map}
    - tupla/lista: (pipeline, feature_cols?, label_names?, thresholds?)
    - pipeline “pelado”: usa .classes_ si existe
    Devuelve dict: {"pipe","cols","labels","thr_map"} o None si no existe.
    """
    import joblib
    p = os.path.join(art_dir, "complete_model_thresholded_bundle.joblib")
    if not os.path.exists(p):
        return None
    obj = joblib.load(p)

    pipe = cols = labels = thr_map = None

    if isinstance(obj, dict):
        pipe   = obj.get("pipeline") or obj.get("pipe") or obj.get("model") or obj.get("estimator")
        cols   = obj.get("feature_cols") or obj.get("X_cols") or obj.get("cols")
        labels = obj.get("label_names") or obj.get("classes")
        thr_map= obj.get("thresholds") or obj.get("thr_map") or obj.get("threshold_map")
    elif isinstance(obj, (tuple, list)):
        if len(obj) >= 1: pipe = obj[0]
        if len(obj) >= 2 and obj[1] is not None:
            cols = list(obj[1]) if not isinstance(obj[1], str) else [obj[1]]
        if len(obj) >= 3 and obj[2] is not None:
            labels = list(obj[2]) if not isinstance(obj[2], str) else [obj[2]]
        if len(obj) >= 4 and isinstance(obj[3], dict):
            thr_map = obj[3]
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
    """
    Prioridad:
    1) bundle (si trae pipeline válido)
    2) estándar: pipeline_ovr_logreg.joblib + feature_cols.csv + label_names.csv (+ thresholds.json)
    """
    import joblib
    b = _try_load_bundle(artifacts_dir)
    if b is not None and b.get("pipe") is not None:
        if b.get("thr_map") is None:
            b["thr_map"] = _load_threshold_map(artifacts_dir)
        return b, None, None

    p = os.path.join(artifacts_dir, "pipeline_ovr_logreg.joblib")
    f = os.path.join(artifacts_dir, "feature_cols.csv")
    l = os.path.join(artifacts_dir, "label_names.csv")
    if not (os.path.exists(p) and os.path.exists(f) and os.path.exists(l)):
        return None, None, None

    return {
        "pipe": joblib.load(p),
        "cols_csv": f,
        "labels_csv": l,
        "thr_map": _load_threshold_map(artifacts_dir)
    }, f, l

def predict_labels(feats: dict, artifacts_dir="artifacts", threshold=0.5):
    """
    Devuelve (labels_activas, scores_dict) o (None, None) si no hay artefactos válidos.
    """
    loaded, _, _ = load_artifacts(artifacts_dir)
    if not loaded or loaded.get("pipe") is None:
        return None, None

    pipe   = loaded["pipe"]
    thr_map= loaded.get("thr_map") or {}

    # Resolver columnas/labels
    if "cols_csv" in loaded:
        cols   = pd.read_csv(loaded["cols_csv"], header=None)[0].tolist()
        labels = pd.read_csv(loaded["labels_csv"], header=None)[0].tolist()
    else:
        cols   = loaded.get("cols") or list(feats.keys())
        labels = loaded.get("labels") or getattr(pipe, "classes_", None)
        if labels is None:
            return None, None

    # Fila ordenada por cols
    row = {c: np.nan for c in cols}
    for k, v in feats.items():
        if k in row: row[k] = v
    X = pd.DataFrame([row])[cols]

    # Probabilidades (o decision_function -> sigmoide)
    if hasattr(pipe, "predict_proba"):
        y = pipe.predict_proba(X)[0]
    elif hasattr(pipe, "decision_function"):
        df = pipe.decision_function(X)
        y = 1/(1+np.exp(-np.asarray(df)[0]))
    else:
        return None, None

    scores = {labels[i]: float(y[i]) for i in range(len(labels))}
    activos = []
    for lbl, s in scores.items():
        thr_lbl = float(thr_map.get(lbl, threshold))
        if s >= thr_lbl:
            activos.append(lbl)
    return activos, scores
