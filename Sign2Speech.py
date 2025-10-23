import cv2
import mediapipe as mp
import pyttsx3
import pickle
import numpy as np
from collections import deque, Counter
import os

# ===============================
# 1️⃣ Define all 20 gestures
# ===============================
gestures = [
    "Hello", "What is your name", "How are you", "I am fine", "Thank you",
    "Good morning", "Good night", "Yes", "No", "I love you",
    "Come here", "Go there", "Please help me", "I need water", "I am hungry",
    "Goodbye", "See you soon", "Nice to meet you", "Where are you", "Take care"
]

# ===============================
# 2️⃣ Load pretrained KNN model
# ===============================
try:
    with open("gesture_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Pretrained model loaded successfully!")
except:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Simulate dataset
    X = np.random.rand(200, 63)
    y = [np.random.choice(gestures) for _ in range(200)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Calculate model accuracy
    y_pred = model.predict(X_test)
    acc = np.round(accuracy_score(y_test, y_pred)*100, 2)
    print(f"⚠️ No real model found; using simulated KNN. Accuracy: {acc}%")

# ===============================
# 3️⃣ Setup TTS
# ===============================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print(f"🔊 Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

# ===============================
# 4️⃣ Mediapipe setup
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ===============================
# 5️⃣ Load gesture images
# ===============================
gesture_images = {}
for g in gestures:
    path = os.path.join("gesture_images", f"{g}.png")  # folder with 20 images
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (200, 200))
        gesture_images[g] = img
    else:
        # Placeholder image
        gesture_images[g] = np.ones((200,200,3), dtype=np.uint8)*200

# ===============================
# 6️⃣ Simple gesture detection
# ===============================
def detect_simple_gesture(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    fingers = []
    for i, tip in enumerate(finger_tips):
        if i == 0:  # thumb
            fingers.append(1 if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x else 0)
        else:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    if fingers == [1, 1, 1, 1, 1]: return "Hello"
    elif fingers == [1, 0, 0, 0, 0]: return "Yes"
    elif fingers == [0, 1, 1, 0, 0]: return "No"
    elif fingers == [1, 1, 0, 0, 1]: return "I love you"
    elif fingers == [0,1,0,0,1]: return "Thank you"
    elif fingers == [1,1,1,0,0]: return "Good morning"
    elif fingers == [0,1,1,1,0]: return "Good night"
    elif fingers == [1,0,1,0,1]: return "Come here"
    elif fingers == [0,0,1,1,1]: return "Go there"
   
    else: return None

# ===============================
# 7️⃣ Webcam loop
# ===============================
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=5)
gesture_active = False

print("🎥 Starting Gesture Recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_name = None

    # Detect hand
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            simple = detect_simple_gesture(hand_landmarks)
            if simple:
                gesture_name = simple
            else:
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                flattened = np.array(landmarks).reshape(1, -1)
                gesture_name = model.predict(flattened)[0]

            if gesture_name:
                buffer.append(gesture_name)

    # Use most frequent gesture in buffer
    if buffer:
        gesture_name = Counter(buffer).most_common(1)[0][0]

    # Speak only once per gesture
    if gesture_name and not gesture_active:
        print(f"🖐️ Detected Gesture: {gesture_name}")
        speak(gesture_name)
        gesture_active = True

    # Reset when hand disappears
    if not result.multi_hand_landmarks:
        gesture_active = False
        buffer.clear()

    # Display gesture text
    cv2.putText(frame, f"Gesture: {gesture_name if gesture_name else '...'}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if gesture_name else (0,0,255), 2)

    # Combine webcam frame + gesture image display
    if gesture_name in gesture_images:
        right_img = gesture_images[gesture_name]
    else:
        right_img = np.ones((200,200,3), dtype=np.uint8)*100

    # Resize right image to match webcam height
    h1, w1, _ = frame.shape
    scale = h1 / right_img.shape[0]
    new_w = int(right_img.shape[1] * scale)
    right_img_resized = cv2.resize(right_img, (new_w, h1))

    combined = np.hstack((frame, right_img_resized))
    cv2.imshow("Gesture Recognition + Visuals", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
