import time
import csv
import cv2
import mediapipe as mp

# import the mediapipe library in order to find the landmark data of hands in images
mp_hands = mp.solutions.hands

# grab the webcam data, as well as initiate a counter to 0 to see how many images are taken
camera_capture = cv2.VideoCapture(1)
counter = 0

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    # using the hands model by mediapipe, grab frames from the camera
    while camera_capture.isOpened():
        success, image = camera_capture.read()
        if not success:
            print("Oops! Missed a frame")
            continue

        # create a window for the webcam, that shows the frames read in from the webcam, but flipped so the video is the 'right' way around
        cv2.imshow("Sign Language Training", cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord("a"):
            # when you press the 'a' key, the counter increments to show how many pictures you have taken
            counter += 1
            print(counter)
            # the program then sleeps for 3 seconds to allow you to do the sign after pressing the button
            time.sleep(3)
            # read in a frame from the camera
            success, image = camera_capture.read()
            if not success:
                print("Oops! Missed a frame")
                continue
            # flip the image and convert it to RGB order rather than BGR, so that mediapipe can process it
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            temp_write_to_csv_left = ""
            temp_write_to_csv_right = ""
            if results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # go through each hand and if the landmark is on the left, add it to the left label, and if its on the right, add it to the right label
                    if results.multi_handedness[idx].classification[0].label.__eq__("Left"):
                        temp_write_to_csv_left = "0"
                        for landmark in hand_landmarks.landmark:
                            temp_write_to_csv_left += ", " + str(landmark.x) + ", " + str(landmark.y)
                    else:
                        temp_write_to_csv_right = "1"
                        for landmark in hand_landmarks.landmark:
                            temp_write_to_csv_right += ", " + str(landmark.x) + ", " + str(landmark.y)
            else:
                print("None")

            # print out the labels so that I can see both hands were recognised
            print("Right:", temp_write_to_csv_right)
            print("Left:", temp_write_to_csv_left)

            # if there was no hand landmarks for a hand, fill the label with 0s
            if temp_write_to_csv_right.__len__() == 0:
                temp_write_to_csv_right = "1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0," \
                                          " 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"
            if temp_write_to_csv_left.__len__() == 0:
                temp_write_to_csv_left = "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, " \
                                         "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"
            # write the labels to a file
            with open("fingerspelling.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow("0, " + temp_write_to_csv_left + ", " + temp_write_to_csv_right)
                # 1 - a, 2 - b, 3 - c, 4 - d, 5 - e, 6 - f, 7 - g, 8 - h, 9 - i, 10 - j, 11 - k, 12 - l, 13 - m, 14 - n
                # 15 - o, 16 - p, 17 - q, 18 - r, 19 - s, 20 - t, 21 - u, 22 - v, 23 - w, 24 - x, 25 - y, 26 - z
                # 27 - next character, 28 - next word

        # close the window when q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
camera_capture.release()
