# Concrete Strength — Streamlit App

A simple Streamlit app that predicts **compressive strength (MPa)** from the classic *concrete* dataset.

## 🧩 Files
- `concrete.csv` — dataset
- `model.joblib`, `metadata.json` — trained model + metadata (already generated)
- `app.py` — Streamlit app
- `train.py` — script to retrain locally if you change anything
- `requirements.txt` — Python deps

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Community Cloud
1. Push these files to a **public GitHub repo** (e.g., `yourname/concrete-streamlit`).
2. Go to **https://share.streamlit.io** → **Deploy an app**.
3. Connect your GitHub, select the repo, branch, and `app.py` as the entry point.
4. Click **Deploy**. First build can take a few minutes.

> Tip: If you want to retrain on Streamlit Cloud instead of shipping `model.joblib`, you can delete the artifacts and add a small training step in `app.py`, but precomputing the model keeps the app fast and reliable.
