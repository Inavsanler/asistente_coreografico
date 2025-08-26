
# 🩰 Asistente Coreográfico 

Uso local:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Streamlit Cloud: usa MediaPipe (CPU). Activa YOLO con `ENABLE_YOLO=true` en entornos GPU.
Artefactos: bundle `complete_model_thresholded_bundle.joblib` o `pipeline_ovr_logreg.joblib` + CSVs en ./artifacts o vía Releases.
