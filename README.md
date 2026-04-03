# Real-time Sign Language Interpreter

Local-first sign language recognition app built with Streamlit, MediaPipe, and scikit-learn.

## Features

- Real-time webcam inference (OpenCV + MediaPipe hand landmarks)
- Stable hold-to-confirm prediction flow
- Top-3 candidate display with confidence
- Sentence builder with clear/undo controls
- Optional text-to-speech on Windows

## Tech Stack

- Streamlit (UI)
- MediaPipe (landmark extraction)
- OpenCV (camera capture)
- scikit-learn (classification model)

## Project Structure

- `app.py`: Main local app
- `train_model.py`: Model training script
- `ensemble.py`: Ensemble wrapper used by training/inference
- `dataset.csv`: Engineered training dataset
- `train.csv`: Source metadata for samples
- `sign_model.pkl`: Saved trained model
- `label_encoder.pkl`: Saved label encoder
- `model_config.txt`: Expected input feature size for inference

## Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## Usage

1. Start the app.
2. Enable `Start camera`.
3. Keep one hand visible in good lighting.
4. Hold a sign until confidence and hold bar confirm it.
5. Build sentence with detected words.

## Retraining

Retrain if you update data or classes:

```powershell
python train_model.py
```

This updates:

- `sign_model.pkl`
- `label_encoder.pkl`
- `model_config.txt`

## Troubleshooting

### Webcam does not open

- Close other apps using camera (Teams, Zoom, browser tabs).
- Disconnect and reconnect webcam.
- Restart Streamlit app.

### Camera opens but prediction is unstable

- Improve lighting and keep hand centered.
- Hold each sign steady for longer.
- Increase hold frames in app settings.

### Feature mismatch / model load issues

- Keep `sign_model.pkl`, `label_encoder.pkl`, and `model_config.txt` in project root.
- Retrain with `python train_model.py` when model artifacts are outdated.

### Port is already in use

```powershell
streamlit run app.py --server.port 8502
```

## Notes

- This repository is optimized for local execution.
- OpenCV camera device access is not reliable on most cloud hosts.
