import cv2
import mediapipe as mp
import time

vid_cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
HAND_LMS = mp_hands.HandLandmark
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = vid_cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)

    # print(results.thumb_tip)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            top_tips = {}
            for point in HAND_LMS:
                normalised_lm = hand_lms.landmark[point]
                top_tips[point] = normalised_lm.y
            top_tip = min(top_tips, key=top_tips.get)
            print(top_tip)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
