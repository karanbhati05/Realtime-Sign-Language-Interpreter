---
title: Realtime Sign Language Interpreter
emoji: "🖐️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Realtime Sign Language Interpreter

This Space runs a Streamlit app for sign-language prediction using MediaPipe landmarks and a trained sklearn model.

## Notes

- The Space runs on a cloud server, so server-side webcam device access (`cv2.VideoCapture(0)`) is not available.
- The app now handles this gracefully and shows a message when webcam is unavailable.
- For live camera demo in cloud, use browser camera streaming integration (for example, streamlit-webrtc).

## Local Run

```bash
streamlit run app.py
```
