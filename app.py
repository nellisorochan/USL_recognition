import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pandas as pd
import queue
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Розпізнавання української дактильної абетки",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_letter' not in st.session_state:
    st.session_state.current_letter = ""
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'last_probs' not in st.session_state:
    st.session_state.last_probs = pd.DataFrame({'Буква': ['-'], 'Ймовірність': [0]})

@st.cache_resource
def get_result_queue():
    return queue.Queue()

@st.cache_resource
def get_frame_buffer():
    return deque(maxlen=25)

@st.cache_resource
def get_live_state():
    return {"letter": "—", "conf": 0.0}

@st.cache_resource
def get_frame_counter():
    return {"count": 0}

frame_buffer = get_frame_buffer()
result_queue = get_result_queue()
live_state = get_live_state()
frame_skip_counter = get_frame_counter()

@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model("sign_language_lstm_model_v11_9759.h5")
        idx_to_letter = {0: 'Є', 1: 'І', 2: 'Ї', 3: 'А', 4: 'Б', 5: 'В', 6: 'Г', 7: 'Д', 8: 'Е', 9: 'Ж', 10: 'З', 11: 'И', 12: 'Й', 13: 'К', 14: 'Л', 15: 'М', 16: 'Н', 17: 'О', 18: 'П', 19: 'Р', 20: 'С', 21: 'Т', 22: 'У', 23: 'Ф', 24: 'Х', 25: 'Ц', 26: 'Ч', 27: 'Ш', 28: 'Щ', 29: 'Ь', 30: 'Ю', 31: 'Я', 32: 'Ґ'}
    except Exception as e:
        st.error(f"Помилка завантаження моделі: {e}")
        model, idx_to_letter = None, {i: chr(1040+i) for i in range(33)}
    
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return model, idx_to_letter, detector

model, idx_to_letter, detector = load_resources()

def add_velocity(seq_array):
    velocity = np.zeros_like(seq_array)
    raw_velocity = seq_array[1:] - seq_array[:-1]
    noise_threshold = 0.015 
    filtered_velocity = np.where(np.abs(raw_velocity) < noise_threshold, 0.0, raw_velocity)
    velocity[1:] = filtered_velocity
    return np.concatenate([seq_array, velocity], axis=-1)

def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        
        # Нормалізація координат
        base = lm.landmark[0]
        coords = []
        for l in lm.landmark:
            coords.extend([l.x - base.x, l.y - base.y, l.z - base.z])
        norm_coords = np.array(coords, dtype=np.float32)
        
        ref_x = lm.landmark[9].x - base.x
        ref_y = lm.landmark[9].y - base.y
        ref_z = lm.landmark[9].z - base.z
        scale = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
        if scale > 1e-6:
            norm_coords = norm_coords / scale
        
        if len(frame_buffer) > 0:
            norm_coords = 0.9 * norm_coords + 0.1 * frame_buffer[-1]

        frame_buffer.append(norm_coords)

        frame_skip_counter["count"] += 1
        if len(frame_buffer) == 25 and frame_skip_counter["count"] % 8 == 0:
            if model:
                seq_np = np.array(list(frame_buffer), dtype=np.float32)
                seq_with_velocity = add_velocity(seq_np)
                inp = np.expand_dims(seq_with_velocity, axis=0)
                
                prediction = model.predict(inp, verbose=0)[0]
                idx = np.argmax(prediction)
                conf = float(prediction[idx])
                
                if conf > 0.82:
                    letter = idx_to_letter.get(idx, "?")
                    live_state["letter"] = letter
                    live_state["conf"] = conf
                    
                    top_idx = np.argsort(prediction)[-5:][::-1]
                    probs_df = pd.DataFrame({
                        'Буква': [idx_to_letter.get(i, "?") for i in top_idx],
                        'Ймовірність': [prediction[i] * 100 for i in top_idx]
                    })
                    
                    result_queue.put({
                        "letter": letter,
                        "conf": conf,
                        "probs": probs_df
                    })
    else:
        frame_buffer.clear()
        live_state["letter"] = "—"
        live_state["conf"] = 0.0

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("""
    <style>
    .stApp { background-color: #F0F2F6; }
    .css-card { background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1rem; }
    .gesture-char { font-size: 110px; font-weight: 800; color: #1E3A8A; text-align: center; line-height: 1; margin: 20px 0; }
    .history-box { display: inline-block; padding: 8px 12px; margin: 2px; background: #E5E7EB; border-radius: 5px; font-weight: bold; color: #374151; }
    .history-box-active { background: #C5A059 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("Розпізнавання української дактильної абетки")
    if st.button("Очистити історію", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_letter = ""
        st.session_state.confidence = 0.0
        st.rerun()
    st.divider()
    st.radio("Меню", ["Розпізнавання", "Абетка"])

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.write("### Відеопотік (Live)")
    webrtc_streamer(
        key="daktyl-v3",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 25},
            "audio": False
        },
        async_processing=True,
    )

with col_right:
    st.write("### Поточний жест")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    display_char = st.session_state.current_letter if st.session_state.current_letter else "—"
    st.markdown(f'<div class="gesture-char">{display_char}</div>', unsafe_allow_html=True)
    
    conf_val = st.session_state.confidence
    st.progress(conf_val)
    st.write(f"Впевненість: **{int(conf_val*100)}%**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="css-card"><b>Історія сесії</b><br><br>', unsafe_allow_html=True)
    if st.session_state.history:
        h_html = ""
        for i, l in enumerate(st.session_state.history[-12:]):
            active = "history-box-active" if i == len(st.session_state.history[-12:])-1 else ""
            h_html += f'<span class="history-box {active}">{l}</span>'
        st.markdown(h_html, unsafe_allow_html=True)
        st.write(f"**СЛОВО:** {''.join(st.session_state.history)}")
    else:
        st.caption("Почніть показувати жести...")
    st.markdown('</div>', unsafe_allow_html=True)

st_autorefresh(interval=1500, key="ui_stable_refresh")

while not result_queue.empty():
    res_data = result_queue.get()
    st.session_state.current_letter = res_data["letter"]
    st.session_state.confidence = res_data["conf"]
    st.session_state.last_probs = res_data["probs"]
    
    new_letter = res_data["letter"]
    if new_letter and new_letter != "":
        if not st.session_state.history or new_letter != st.session_state.history[-1]:
            st.session_state.history.append(new_letter)
