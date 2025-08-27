import tempfile, os, streamlit as st, numpy as np, pandas as pd

# usa SOLO init_artifacts (ya hace release + fallback ./artifacts)
from src.artifacts_io import init_artifacts
from src.backends import mediapipe_video_to_keypoints
from src.features import features_coreograficos
from src.suggestions import sugerencias_reglas
from src.model import predict_labels

st.set_page_config(page_title="Asistente Coreográfico", layout="wide")
st.title("🩰 Asistente Coreográfico (MediaPipe + YOLO)")

# --- Secrets / Artifacts init (primero intenta Release, si falla usa ./artifacts)
ENABLE_YOLO = bool(st.secrets.get("ENABLE_YOLO", False))
owner = st.secrets.get("GH_OWNER", None)
repo  = st.secrets.get("GH_REPO", None)
tag   = st.secrets.get("ARTIFACTS_RELEASE_TAG", None)

info = init_artifacts(owner=owner, repo=repo, tag=tag, target_dir="artifacts")

# Mensajes de estado
if info["release_attempted"]:
    if info["variant"]:
        st.caption(f"✅ Artefactos listos (intentado Release {owner}/{repo}@{tag} y detectado {', '.join(info['variant'])}).")
    else:
        st.warning("⚠️ Intenté descargar artefactos del Release, pero no quedaron válidos. Se usará ./artifacts si existen.")
elif info["variant"]:
    st.caption(f"✅ Artefactos locales en ./artifacts: {', '.join(info['variant'])}")
else:
    st.info("ℹ️ No hay secrets de Release y no encontré artefactos en ./artifacts.")

ADV_AVAILABLE = bool(info["variant"])

# --- Sidebar (AQUÍ se definen backend y stride)
with st.sidebar:
    st.header("⚙️ Backend de pose")
    options = ["MediaPipe (CPU)"]
    if ENABLE_YOLO:
        options.append("YOLOv8-Pose (GPU recomendada)")
    backend = st.selectbox("Backend", options)
    stride = st.number_input("Procesar 1 de cada N frames", 1, 10, 2, 1)

    st.divider()
    st.header("🪟 Ventanas")
    win_s = st.slider("Tamaño ventana (s)", 2.0, 10.0, 5.0, 0.5)
    hop_s = st.slider("Salto entre ventanas (s)", 0.5, 5.0, 2.5, 0.5)

    st.divider()
    st.header("🧠 Modelo avanzado (sklearn)")
    use_adv = st.checkbox("Usar artefactos en ./artifacts", value=ADV_AVAILABLE, disabled=not ADV_AVAILABLE)
    thr = st.slider("Umbral global de activación", 0.05, 0.95, 0.50, 0.05)

st.markdown("Sube un **vídeo corto (10–30s)**. En Streamlit Cloud se recomienda **MediaPipe (CPU)**.")

def run_backend(path, backend, stride):
    if backend.startswith("MediaPipe"):
        return mediapipe_video_to_keypoints(path, sample_stride=int(stride))
    else:
        try:
            from src.backends import yolo_video_to_keypoints
        except Exception as e:
            st.error("YOLOv8-Pose no está disponible en este entorno. Actívalo solo en GPU (ENABLE_YOLO=true) e instala 'ultralytics'.")
            st.stop()
        return yolo_video_to_keypoints(path, model_name="yolov8n-pose.pt", conf=0.25, iou=0.5, stride=int(stride))

def iter_windows(T, fps, win_s, hop_s):
    win = max(1, int(round(win_s * fps)))
    hop = max(1, int(round(hop_s * fps)))
    for s in range(0, T-1, hop):
        e = min(T, s+win)
        if e-s < max(8, int(0.25*win)): break
        yield s, e

video = st.file_uploader("🎬 Sube tu vídeo", type=["mp4","mov","m4v","avi","mkv"])
if video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.read())
        vpath = tmp.name
    st.info("Extrayendo poses…", icon="⏳")
    K, fps = run_backend(vpath, backend, stride)
    st.success(f"Keypoints: {K.shape} | FPS≈{fps:.1f}")

    rows=[]; timeline=[]
    T=len(K)
    for s,e in iter_windows(T, fps, win_s, hop_s):
        feats = features_coreograficos(K[s:e], fps=fps)
        sugs = sugerencias_reglas(feats)
        labels=None; scores=None
        if use_adv:
            labels, scores = predict_labels(feats, artifacts_dir="artifacts", threshold=float(thr))
        rows.append({"start_f":s, "end_f":e, "start_s":s/fps, "end_s":e/fps, **feats, "labels":"|".join(labels or [])})
        timeline.append({"start":s/fps, "end":e/fps, "sugerencias":sugs, "labels":labels})

    df = pd.DataFrame(rows)
    st.subheader("⏱️ Timeline de ventanas")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="timeline.csv", mime="text/csv")

    st.subheader("✨ Sugerencias por ventana")
    for i,t in enumerate(timeline,1):
        st.markdown(f"**Ventana {i} — {t['start']:.2f}s–{t['end']:.2f}s**")
        for s in t["sugerencias"]:
            st.write(f"- {s}")
        if use_adv and t["labels"] is not None:
            st.write(f"Etiquetas: {', '.join(t['labels']) if t['labels'] else '—'}")
else:
    st.info("Sube un vídeo para comenzar.", icon="🎥")

