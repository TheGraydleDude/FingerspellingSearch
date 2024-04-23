import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2

# changed for privacy
model_loc = ""
new_model = tf.keras.models.load_model(model_loc,
                                       compile=False)
new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture("C:\\Users\\ausaf\\Documents\\Computing NEA\\Main\\video1.avi")
video_frames = []
while True:
    success, frame = capture.read()

    if success:
        video_frames.append(frame)
    else:
        break

capture.release()

categories = [",", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
              "u", "v", "w", "x", "y", "z", ";", ":"]
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(video_frames):
        image = cv2.flip(video_frames[idx], 1)

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        left = []
        right = []
        if results.multi_handedness:
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if results.multi_handedness[index].classification[0].label.__eq__("Left"):
                    for landmark in hand_landmarks.landmark:
                        left.append(landmark.x)
                        left.append(landmark.y)
                else:
                    for landmark in hand_landmarks.landmark:
                        right.append(landmark.x)
                        right.append(landmark.y)
        if right.__len__() == 0:
            right = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0]
        if left.__len__() == 0:
            left = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0]
        right = left + right
        if right.__len__() != 84:
            continue
        prediction = new_model(np.array(right).reshape(1, 84))
        predict_class = np.argmax(prediction[0])
        print(categories[predict_class])
