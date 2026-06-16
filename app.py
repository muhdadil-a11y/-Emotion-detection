import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load models
emotion_model = load_model("models/emotion_model.h5")
face_detector = YOLO("yolov8n.pt")

emotion_labels = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

st.title("😄 Real-Time Emotion Detection")
st.write("Using YOLOv8 + CNN (FER Classification)")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect faces
        results = face_detector(img, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48,48))
                normalized = resized / 255.0
                reshaped = normalized.reshape(1,48,48,1)

                pred = emotion_model.predict(reshaped, verbose=0)
                emotion = emotion_labels[np.argmax(pred)]

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    img,
                    emotion,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2
                )

        return img

webrtc_streamer(
    key="emotion-detection-stream",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
)
