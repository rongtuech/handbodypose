import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands


class HandPoseDetector:
    def __init__(self, num_hands = 1):
        self.hands = mp_hands.Hands(static_image_mode = False,
                                    max_num_hands=num_hands,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    def __call__(self, image):
        iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # draw on input image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return results.multi_hand_landmarks[0],results.multi_handedness[0]

    def __delete__(self):
        self.hands.close()
        print("release mem for hand processor")