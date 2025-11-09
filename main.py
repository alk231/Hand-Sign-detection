import cv2
import numpy as np
from keras.models import load_model

model = load_model("E:/open cv/SignDetcection/signLanguageModel_AB.keras")
labels = [
    'Blank','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'
]
 # must match training

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (0,40), (300,300), (255,255,255), 2)
    crop = frame[40:300, 0:300]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48,48))
    input_data = extract_features(gray)

    pred = model.predict(input_data)
    pred_label = labels[pred.argmax()]
    accu = "{:.2f}".format(np.max(pred)*100)

    cv2.putText(frame, f"{pred_label} {accu}%", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
