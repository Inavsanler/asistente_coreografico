# src/backends.py
import numpy as np

def mediapipe_video_to_keypoints(video_path, sample_stride=2):
    try:
        import cv2
    except Exception as e:
        raise ImportError(
            "OpenCV no disponible. Asegúrate de tener 'opencv-python-headless' en requirements "
            "y Python 3.12/3.11 en Streamlit Cloud."
        ) from e
    try:
        import mediapipe as mp
    except Exception as e:
        raise ImportError(
            "MediaPipe no disponible o incompatible. Usa mediapipe==0.10.21 y Python 3.12/3.11."
        ) from e

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    frames=[]; idx=0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if (idx % max(1, int(sample_stride))) != 0:
                idx+=1; continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lmks = res.pose_landmarks.landmark
                xy = np.array([[l.x, l.y] for l in lmks], dtype=np.float32)
            else:
                xy = np.full((33,2), np.nan, dtype=np.float32)
            frames.append(xy); idx+=1
    finally:
        cap.release(); pose.close()
    return np.asarray(frames, dtype=np.float32), float(fps)
