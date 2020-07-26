from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model('./best_model.h5')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    "../Computer-Vision-with-Python/DATA/haarcascades/haarcascade_frontalface_default.xml")
while True:
    ret, frame = cap.read()
    img_cpy = frame.copy()
    faces = face_cascade.detectMultiScale(img_cpy, 1.2, 5)
    for (x, y, w, h) in faces:
        roi = img_cpy[y - 10:y + 10 + h, x - 10:x + 10 + w]
        roi = cv2.resize(roi, (224, 224))
        roi = preprocess_input(roi)
        roi = np.expand_dims(roi, axis=0)
        ans = model.predict(roi)
        if ans < 0.5:
            cv2.rectangle(img_cpy, (x - 10, y), (x + w + 10, y + h + 10), (0, 255, 0), 5)
            cv2.putText(img_cpy, "Mask", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        else:
            cv2.rectangle(img_cpy, (x - 10, y), (x + w + 10, y + h + 10), (0, 0, 255), 5)
            cv2.putText(img_cpy, "No Mask", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Face", img_cpy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
