import os
import platform
import subprocess
import warnings
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

st.set_page_config(page_title="Sign Language Interpreter", layout="wide")


@st.cache_resource
def load_model_assets():
    model = joblib.load("sign_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    with open("model_config.txt", "r", encoding="utf-8") as f:
        n_features = int(f.read().strip())
    return model, label_encoder, n_features


@st.cache_resource
def load_hands_detector():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    return mp_hands, mp_draw, mp_style, hands


def speak(text):
    if platform.system() != "Windows":
        return
    try:
        subprocess.Popen(
            [
                "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "-Command",
                (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$s.Rate = 1; "
                    f"$s.Speak(\"{text}\")"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# Feature engineering must match train_model.py.
fingertips = [4, 8, 12, 16, 20]
finger_joints = [(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20), (4, 12), (8, 20)]


def angle_single(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))


def extract_features_from_buffer(frames_list):
    frames_arr = np.array(frames_list)

    mean_f = frames_arr.mean(axis=0)
    std_f = frames_arr.std(axis=0)
    motion = frames_arr[-1] - frames_arr[0]
    mid_f = frames_arr[len(frames_arr) // 2]

    mean_r = mean_f.reshape(21, 3)
    wrist = mean_r[0]
    tip_dists = np.array([np.linalg.norm(mean_r[t] - wrist) for t in fingertips])
    curl_ang = np.array([angle_single(mean_r[a], mean_r[b], mean_r[c]) for a, b, c in finger_joints])
    inter = np.array([np.linalg.norm(mean_r[p] - mean_r[q]) for p, q in pairs])
    palm_w = np.linalg.norm(mean_r[5] - mean_r[17])
    palm_h = np.linalg.norm(mean_r[0] - mean_r[9])
    spread = mean_r[fingertips, :].std(axis=0)

    return np.concatenate([
        mean_f,
        std_f,
        motion,
        mid_f,
        tip_dists,
        curl_ang,
        inter,
        [palm_w, palm_h],
        spread,
    ])


def init_state():
    if "sentence" not in st.session_state:
        st.session_state.sentence = []
    if "last_added" not in st.session_state:
        st.session_state.last_added = ""
    if "landmark_buffer" not in st.session_state:
        st.session_state.landmark_buffer = deque(maxlen=15)
    if "hold_buffer" not in st.session_state:
        st.session_state.hold_buffer = deque(maxlen=20)
    if "live_label" not in st.session_state:
        st.session_state.live_label = ""
    if "live_confidence" not in st.session_state:
        st.session_state.live_confidence = 0.0
    if "live_top3" not in st.session_state:
        st.session_state.live_top3 = []
    if "camera" not in st.session_state:
        st.session_state.camera = None
    if "failed_reads" not in st.session_state:
        st.session_state.failed_reads = 0


def open_camera():
    if st.session_state.camera is not None:
        return st.session_state.camera

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None

    st.session_state.camera = cap
    st.session_state.failed_reads = 0
    return cap


def close_camera():
    cap = st.session_state.camera
    if cap is not None:
        cap.release()
    st.session_state.camera = None
    st.session_state.failed_reads = 0


init_state()

try:
    model, le, n_features = load_model_assets()
    mp_hands, mp_draw, mp_style, hands_detector = load_hands_detector()
except Exception as e:
    st.error(f"Failed to initialize app assets: {e}")
    st.stop()

st.title("Real-time Sign Language Interpreter")
st.caption("Run locally and click Start camera for live recognition")

col_vid, col_ctrl = st.columns([3, 2])

with col_ctrl:
    st.subheader("Settings")
    confidence_threshold = st.slider("Confidence threshold", 0.50, 0.99, 0.80, 0.01)
    hold_frames = st.slider("Hold frames to confirm", 5, 40, 20)
    voice_on = st.checkbox("Voice output", value=True)
    speak_mode = st.radio("Speak mode", ["Each word", "Full sentence"], horizontal=True)

    st.divider()
    st.subheader("Sentence")
    sentence_box = st.empty()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Clear", use_container_width=True):
            st.session_state.sentence = []
            st.session_state.last_added = ""
    with b2:
        if st.button("Undo", use_container_width=True):
            if st.session_state.sentence:
                st.session_state.sentence.pop()
                st.session_state.last_added = (
                    st.session_state.sentence[-1] if st.session_state.sentence else ""
                )
    with b3:
        if st.button("Speak", use_container_width=True) and st.session_state.sentence:
            speak(" ".join(st.session_state.sentence))

    st.divider()
    st.subheader("Live prediction")
    pred_box = st.empty()
    conf_bar = st.empty()

    st.subheader("Top 3 candidates")
    top3_box = st.empty()

with col_vid:
    st.toggle("Start camera", key="run_camera", value=False)
    frame_window = st.empty()

if st.session_state.hold_buffer.maxlen != hold_frames:
    st.session_state.hold_buffer = deque(list(st.session_state.hold_buffer), maxlen=hold_frames)

@st.fragment(run_every="60ms")
def render_live_section():
    run_camera = st.session_state.get("run_camera", False)

    if run_camera:
        cap = open_camera()
        if cap is None:
            frame_window.error("Could not access webcam. Close other apps using camera and retry.")
        else:
            ok, frame = cap.read()
            if not ok:
                st.session_state.failed_reads += 1
                if st.session_state.failed_reads >= 15:
                    frame_window.error("Could not read webcam frame consistently. Please restart camera.")
                    close_camera()
                else:
                    frame_window.warning("Temporary camera read issue. Retrying...")
            else:
                st.session_state.failed_reads = 0
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(rgb)

                label = ""
                confidence = 0.0
                top3 = []

                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )

                    lm_raw = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark]).flatten()
                    st.session_state.landmark_buffer.append(lm_raw)

                    if len(st.session_state.landmark_buffer) == 15:
                        features = extract_features_from_buffer(list(st.session_state.landmark_buffer))
                        if len(features) == n_features:
                            proba = model.predict_proba([features])[0]
                            top_idx = int(np.argmax(proba))
                            confidence = float(proba[top_idx])
                            label = str(le.classes_[top_idx])
                            top3_idx = proba.argsort()[-3:][::-1]
                            top3 = [(str(le.classes_[i]), float(proba[i])) for i in top3_idx]

                            st.session_state.hold_buffer.append(label)
                            if (
                                confidence >= confidence_threshold
                                and len(st.session_state.hold_buffer) == hold_frames
                                and len(set(st.session_state.hold_buffer)) == 1
                                and label != st.session_state.last_added
                            ):
                                st.session_state.sentence.append(label)
                                st.session_state.last_added = label
                                if voice_on and speak_mode == "Each word":
                                    speak(label)
                else:
                    st.session_state.hold_buffer.clear()
                    st.session_state.landmark_buffer.clear()

                st.session_state.live_label = label
                st.session_state.live_confidence = confidence
                st.session_state.live_top3 = top3

                color = (0, 200, 100) if confidence >= confidence_threshold else (0, 140, 255)
                cv2.rectangle(frame, (10, 50), (240, 70), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 50), (10 + int(confidence * 230), 70), color, -1)
                cv2.putText(
                    frame,
                    f"{label}  {confidence * 100:.0f}%",
                    (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )

                filled = sum(1 for x in st.session_state.hold_buffer if x == label) if label else 0
                cv2.rectangle(frame, (10, 76), (240, 88), (40, 40, 40), -1)
                if hold_frames > 0:
                    cv2.rectangle(
                        frame,
                        (10, 76),
                        (10 + int(filled / hold_frames * 230), 88),
                        (255, 200, 0),
                        -1,
                    )

                frame_window.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
    else:
        close_camera()
        frame_window.info("Camera is stopped. Enable Start camera to begin live recognition.")

    sentence_text = " ".join(st.session_state.sentence)
    sentence_box.markdown(f"### {sentence_text}" if sentence_text else "*Start signing...*")

    live_label = st.session_state.live_label
    live_confidence = st.session_state.live_confidence
    live_top3 = st.session_state.live_top3
    pred_box.markdown(f"**`{live_label}`**" if live_label else "*No hand detected*")
    conf_bar.progress(float(round(live_confidence, 2)), text=f"Confidence: {live_confidence * 100:.0f}%")

    if live_top3:
        top3_box.markdown(
            "\n".join(
                [f"**{i+1}.** `{sign}` - {prob * 100:.0f}%" for i, (sign, prob) in enumerate(live_top3)]
            )
        )
    else:
        top3_box.markdown("")


render_live_section()

if voice_on and speak_mode == "Full sentence" and st.session_state.sentence:
    if st.button("Speak full sentence"):
        speak(" ".join(st.session_state.sentence))
