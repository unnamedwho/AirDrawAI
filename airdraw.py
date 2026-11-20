
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime

# Load trained YOLO model
model = YOLO("models/best.pt")

# MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

canvas = None
previous_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Recognize gesture with YOLO
    result = model.predict(frame, conf=0.4, verbose=False)[0]

    gesture = None
    if len(result.boxes) > 0:
        best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
        gesture = model.names[int(best_box.cls[0])]

    # Track index fingertip
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    fingertip = None

    if hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0]
        tip = hand.landmark[8]
        fingertip = (int(tip.x * width), int(tip.y * height))

    # Gesture controls
    if gesture == "point" and fingertip:
        if previous_point is not None:
            cv2.line(canvas, previous_point, fingertip, (255, 0, 255), 5)
        previous_point = fingertip

    else:
        previous_point = None

    if gesture == "open-palm":
        canvas[:] = 0

    if gesture == "thumbs-up":
        filename = f"airdraw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, canvas)

    # Display drawing over camera feed
    display = cv2.addWeighted(frame, 1, canvas, 0.8, 0)

    cv2.putText(
        display,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("AirDraw AI", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
