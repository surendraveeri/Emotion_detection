import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ensure channels_last
tf.keras.backend.set_image_data_format('channels_last')

# ----------------------------
# Build architecture that exactly matches model_emotion_full.h5 weights
# ----------------------------
def build_emotion_model():
    model = models.Sequential(name="emotion_cnn")

    # conv2d  -> conv2d_1  (64 filters)
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same',
                            input_shape=(48,48,1), name="conv2d"))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same',
                            name="conv2d_1"))
    model.add(layers.MaxPooling2D(pool_size=(3,3), name="max_pooling2d"))
    model.add(layers.Dropout(0.25, name="dropout"))

    # conv2d_2 (128 filters)
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same',
                            name="conv2d_2"))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_1"))
    model.add(layers.Dropout(0.25, name="dropout_1"))

    # conv2d_3 (128 filters)
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same',
                            name="conv2d_3"))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_2"))
    model.add(layers.Dropout(0.25, name="dropout_2"))

    # flatten -> dense -> dense_1 -> dense_2
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(512, activation='relu', name="dense"))
    model.add(layers.Dropout(0.5, name="dropout_3"))
    model.add(layers.Dense(256, activation='relu', name="dense_1"))
    model.add(layers.Dropout(0.5, name="dropout_4"))
    model.add(layers.Dense(7, activation='softmax', name="dense_2"))

    return model


# ----------------------------
# Load weights
# ----------------------------
weights_path = "model_emotion_full.h5"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model file not found: {weights_path}")

model = build_emotion_model()

try:
    model.load_weights(weights_path)
    print("✅ Model weights loaded successfully! Architecture matched.")
except Exception as e:
    print("❌ Error loading weights:", e)
    raise

# ----------------------------
# Labels and colors
# ----------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Sad': (255, 0, 0),
    'Surprise': (255, 255, 0),
    'Neutral': (200, 200, 200)
}

# ----------------------------
# Webcam + face detection
# ----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (cv2.VideoCapture failed)")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

print("🎥 Webcam started — press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Cannot access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))  # shape (1,48,48,1)

        preds = model.predict(roi, verbose=0)
        idx = int(np.argmax(preds))
        emotion = emotion_labels[idx]
        confidence = float(np.max(preds))

        color = emotion_colors[emotion]
        label = f"{emotion} ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

    cv2.imshow("Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Application closed.")
