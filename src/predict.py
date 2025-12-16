import cv2
import numpy as np
import tensorflow as tf
from voice import speak

MODEL_PATH = "model/my_face_model.h5"
LABELS = ["others", "raman"]

model = tf.keras.models.load_model(MODEL_PATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (100,100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    class_id = np.argmax(pred)
    conf = pred[class_id]

    name = LABELS[class_id]
    text = f"{name} ({conf*100:.1f}%)"

    cv2.putText(frame, text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if class_id == 1 and conf > 0.85:
        speak("Welcome boss")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()