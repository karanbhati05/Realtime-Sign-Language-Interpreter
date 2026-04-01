import streamlit as st
import cv2
import os
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
import mediapipe as mp
import joblib
import numpy as np
import platform
from collections import deque
from ensemble import EnsembleModel
import queue

st.set_page_config(
    page_title="Sign Language Interpreter",
    page_icon="👋",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("sign_model.pkl")
    le    = joblib.load("label_encoder.pkl")
    with open("model_config.txt") as f:
        n_features = int(f.read().strip())
    return model, le, n_features

landmark_buffer = deque(maxlen=15)

try:
    model, le, N_FEATURES = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def speak(text):
    import subprocess
    if platform.system() != "Windows":
        return
    try:
        subprocess.Popen(
            ['C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe', '-Command',
             f'Add-Type -AssemblyName System.Speech; '
             f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
             f'$s.Rate = 1; '
             f'$s.Speak("{text}")'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        # Ignore TTS errors so cloud deployment keeps running.
        pass

# Feature engineering — must match train_model.py exactly
fingertips    = [4, 8, 12, 16, 20]
finger_joints = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
pairs         = [(4,8),(8,12),(12,16),(16,20),(4,20),(4,12),(8,20)]

def angle_single(a, b, c):
    ba = a - b; bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))

def extract_features_from_buffer(frames_list):
    frames_arr = np.array(frames_list)  # (15, 63)

    mean_f  = frames_arr.mean(axis=0)          # 63
    std_f   = frames_arr.std(axis=0)           # 63
    motion  = frames_arr[-1] - frames_arr[0]   # 63
    mid_f   = frames_arr[len(frames_arr)//2]   # 63

    mean_r     = mean_f.reshape(21, 3)
    wrist      = mean_r[0]
    tip_dists  = np.array([np.linalg.norm(mean_r[t] - wrist) for t in fingertips])
    curl_ang   = np.array([angle_single(mean_r[a], mean_r[b], mean_r[c]) for a,b,c in finger_joints])
    inter      = np.array([np.linalg.norm(mean_r[p] - mean_r[q]) for p,q in pairs])
    palm_w     = np.linalg.norm(mean_r[5]  - mean_r[17])
    palm_h     = np.linalg.norm(mean_r[0]  - mean_r[9])
    spread     = mean_r[fingertips, :].std(axis=0)

    return np.concatenate([
        mean_f, std_f, motion, mid_f,
        tip_dists, curl_ang, inter,
        [palm_w, palm_h], spread
    ])

# Session state
if "sentence"   not in st.session_state: st.session_state.sentence   = []
if "last_added" not in st.session_state: st.session_state.last_added = ""
if "landmark_buffer" not in st.session_state: st.session_state.landmark_buffer = deque(maxlen=15)
if "hold_buffer" not in st.session_state: st.session_state.hold_buffer = deque(maxlen=20)
if "live_label" not in st.session_state: st.session_state.live_label = ""
if "live_confidence" not in st.session_state: st.session_state.live_confidence = 0.0
if "live_top3" not in st.session_state: st.session_state.live_top3 = []

# UI
st.markdown("## Real-time sign language interpreter")
st.caption("Hold a sign steady — the yellow bar fills as it recognises your gesture")

col_vid, col_ctrl = st.columns([3, 2])

with col_ctrl:
    st.markdown("#### Settings")
    conf_thresh = st.slider("Confidence threshold", 0.50, 0.99, 0.80, 0.01)
    hold_frames = st.slider("Hold frames to confirm", 5, 40, 20)
    voice_on    = st.checkbox("Voice output", value=True)
    speak_mode  = st.radio("Speak mode", ["Each word", "Full sentence"], horizontal=True)

    st.markdown("---")
    st.markdown("#### Sentence")
    sentence_box = st.empty()
    sentence_box.markdown("*Start signing...*")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clear", use_container_width=True):
            st.session_state.sentence   = []
            st.session_state.last_added = ""
    with c2:
        if st.button("Undo", use_container_width=True):
            if st.session_state.sentence:
                st.session_state.sentence.pop()
                st.session_state.last_added = st.session_state.sentence[-1] if st.session_state.sentence else ""
    with c3:
        if st.button("Speak ↗", use_container_width=True):
            if st.session_state.sentence:
                speak(" ".join(st.session_state.sentence))

    st.markdown("---")
    st.markdown("#### Live prediction")
    pred_box = st.empty()
    conf_bar = st.empty()

    st.markdown("#### Top 3 candidates")
    top3_box = st.empty()

with col_vid:
    run          = st.checkbox("Start camera", value=False)
    frame_window = st.empty()

if st.session_state.hold_buffer.maxlen != hold_frames:
    st.session_state.hold_buffer = deque(list(st.session_state.hold_buffer), maxlen=hold_frames)

if run:
    camera_frame = st.camera_input("Capture sign frame")
    if camera_frame is not None:
        bytes_data = camera_frame.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)

        hands_det = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_det.process(rgb)
        hands_det.close()

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
                feats = extract_features_from_buffer(list(st.session_state.landmark_buffer))
                if len(feats) == N_FEATURES:
                    proba = model.predict_proba([feats])[0]
                    top_idx = proba.argmax()
                    confidence = proba[top_idx]
                    label = le.classes_[top_idx]
                    top3_idx = proba.argsort()[-3:][::-1]
                    top3 = [(le.classes_[i], proba[i]) for i in top3_idx]

                    st.session_state.hold_buffer.append(label)
                    if (
                        confidence >= conf_thresh
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

        color = (0, 200, 100) if confidence >= conf_thresh else (0, 140, 255)
        cv2.rectangle(frame, (10, 50), (240, 70), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 50), (10 + int(confidence * 230), 70), color, -1)
        cv2.putText(
            frame,
            f"{label}  {confidence*100:.0f}%",
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

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
    else:
        st.info("Use the camera capture button to provide frames for prediction.")

sent = " ".join(st.session_state.sentence)
sentence_box.markdown(f"### {sent}" if sent else "*Start signing...*")
label = st.session_state.live_label
confidence = st.session_state.live_confidence
top3 = st.session_state.live_top3
pred_box.markdown(f"**`{label}`**" if label else "*No hand detected*")
conf_bar.progress(float(round(confidence, 2)), text=f"Confidence: {confidence*100:.0f}%")

if top3:
    top3_box.markdown("\n".join([
        f"**{i+1}.** `{sign}` — {prob*100:.0f}%"
        for i, (sign, prob) in enumerate(top3)
    ]))
