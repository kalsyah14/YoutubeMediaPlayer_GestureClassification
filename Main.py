#Local Inference (Control Youtube Media Player with Gesture Classification)
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import time

NUM_FRAMES = 30
NUM_LANDMARKS = 21 * 3
CLASS_MAPPING = {0: 'Stop', 1: 'Thumbs Down', 2: 'Thumbs Up'}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
draw = mp.solutions.drawing_utils

model = Sequential([
    LSTM(128, return_sequences=True, activation='tanh', input_shape=(NUM_FRAMES, NUM_LANDMARKS)),
    Dropout(0.5),
    BatchNormalization(),

    LSTM(64, return_sequences=False, activation='tanh'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
model.load_weights('Best_3 Gesture Mediapipe LSTM.h5')

cap = cv2.VideoCapture(1)
sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    class_name = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        sequence.append(keypoints)
        if len(sequence) > NUM_FRAMES:
            sequence.pop(0)

        draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(sequence) == NUM_FRAMES:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            class_name = CLASS_MAPPING[class_id]

            #Youtube Control
            if class_name == "Stop":
                pyautogui.press('k')  # Play/Pause
                time.sleep(0.9)
            elif class_name == "Thumbs Up":
                pyautogui.press("volumeup")  # Volume Up
                time.sleep(0.2)
            elif class_name == "Thumbs Down":
                pyautogui.press("volumedown")  # Volume Down
                time.sleep(0.2)

    #Display
    if class_name:
        cv2.putText(frame, f"Gesture: {class_name}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Youtube Media Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
