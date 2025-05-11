import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from googletrans import Translator
import tempfile
import os

model = load_model("isl_video_model.h5")

gesture_dict = {0: "Call", 1: "Doctor", 2: "Help"}
translator = Translator()

def preprocess_frame(frame, target_size=(128, 128)):
    resized = cv2.resize(frame, target_size)
    return resized / 255.0

def preprocess_video(path, target_frames=30):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened() and len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))

    cap.release()
    while len(frames) < target_frames:
        frames.append(frames[-1]) 

    return np.expand_dims(np.array(frames), axis=0)

def predict_gesture(video_path):
    input_data = preprocess_video(video_path)
    predictions = model.predict(input_data)
    confidence = np.max(predictions)
    class_index = np.argmax(predictions)

    if confidence >= 0.60:
        label = gesture_dict.get(class_index, "Unknown")
        trans_kn = translator.translate(label, src="en", dest="kn").text
        trans_hi = translator.translate(label, src="en", dest="hi").text
        return label, trans_kn, trans_hi, confidence
    else:
        return "Low Confidence", "", "", confidence
st.title("Indian Sign Language To Multilingual Texts")

uploaded_video = st.file_uploader("Upload a gesture video", type=["mp4", "avi", "mov"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    st.video(video_path)
    
    if st.button("Predict Gesture"):
        label, kn, hi, conf = predict_gesture(video_path)
        st.success(f"Predicted Gesture: {label} (Confidence: {conf:.2f})")
        if kn:
            st.write(f"Kannada: {kn}")
            st.write(f"Hindi: {hi}")
