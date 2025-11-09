import cv2
import numpy as np
import os
import string

# Create the directory structure safely
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

for i in range(3):
    os.makedirs(f"data/train/{i}", exist_ok=True)
    os.makedirs(f"data/test/{i}", exist_ok=True)

for i in string.ascii_uppercase:
    os.makedirs(f"data/train/{i}", exist_ok=True)
    os.makedirs(f"data/test/{i}", exist_ok=True)

os.makedirs("data/train/BLANK", exist_ok=True)
os.makedirs("data/test/BLANK", exist_ok=True)

# Mode and settings
mode = 'train'
directory = f"data/{mode}/"
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Count existing images
    count = {i: len(os.listdir(os.path.join(directory, i))) for i in
             [str(x) for x in range(3)] + list(string.ascii_uppercase) + ["BLANK"]}

    # Show image counts
    y_offset = 60
    for key in count.keys():
        cv2.putText(frame, f"{key} : {count[key]}", (10, y_offset),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        y_offset += 12

    # ROI (Region of Interest)
    x1, y1, x2, y2 = 220, 10, 620, 410
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = frame[y1:y2, x1:x2]

    # Processing ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    _, test_image = cv2.threshold(th3, minValue, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("Frame", frame)
    cv2.imshow("test", test_image)

    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == 27:  # ESC key
        break

    # Numbers
    for num in ['0', '1', '2']:
        if interrupt & 0xFF == ord(num):
            cv2.imwrite(f"{directory}{num}/{count[num]}.jpg", roi)

    # Alphabets
    for ch in string.ascii_lowercase:
        if interrupt & 0xFF == ord(ch):
            folder = ch.upper()
            cv2.imwrite(f"{directory}{folder}/{count[folder]}.jpg", roi)

    # Backspace → Save blank image
    if interrupt & 0xFF == 8:
        cv2.imwrite(f"{directory}BLANK/{count['BLANK']}.jpg", roi)

cap.release()
cv2.destroyAllWindows()
