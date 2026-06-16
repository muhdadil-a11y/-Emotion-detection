import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Load trained emotion model
emotion_model = load_model("models/emotion_model.h5")

# Load YOLO face detector
face_detector = YOLO("yolov8n.pt")

# Emotion labels (MUST MATCH class_indices)
emotion_labels = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    results = face_detector(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess face
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48,48))
            normalized = resized / 255.0
            reshaped = normalized.reshape(1,48,48,1)

            # Predict emotion
            preds = emotion_model.predict(reshaped, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]

            # Draw results
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, emotion, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,255,0), 2)

    cv2.imshow("Emotion Detection (YOLO + CNN)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
