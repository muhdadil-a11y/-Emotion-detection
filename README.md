## How It Works

The Emotion Detection System combines face detection and deep learning-based emotion recognition to analyze facial expressions in real time.

### Components Used

- **YOLOv8** – Detects faces in the video stream.
- **FER2013 Dataset** – Used to train the emotion recognition model.
- **TensorFlow/Keras** – Builds and trains the deep learning model.
- **OpenCV** – Captures webcam video and displays results.
- **NumPy** – Handles image and numerical data processing.
- **Python** – Core programming language used for development.

### Workflow

1. The webcam captures live video frames.
2. YOLOv8 processes each frame and detects faces.
3. The detected face region is extracted from the frame.
4. The face image is preprocessed (resized, normalized, and converted into the required format).
5. The trained emotion recognition model analyzes facial features.
6. The model predicts one of the supported emotions:
   - Angry
   - Disgust
   - Fear
   - Happy
   - Sad
   - Surprise
   - Neutral
7. The predicted emotion is displayed above the detected face in real time.
8. The process repeats continuously for every frame captured by the webcam.

This combination of computer vision and deep learning enables fast and accurate real-time emotion recognition from facial expressions.
