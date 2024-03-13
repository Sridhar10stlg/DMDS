import face_recognition
import time
from scipy.spatial import distance as dist
import playsound
from threading import Thread
import cv2
from deepface import DeepFace
import numpy as np

faceCascade = cv2.CascadeClassifier("D:\DMDS\Integrated\DeepFace\deepface.xml")

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive frames
# the eye must be below the threshold to set off the alarm
MIN_AER = 0.25
MINI_AER = 0.15
EYE_AR_CONSEC_FRAMES = 5

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def sound_alarm(alarm_file):
    # play an alarm sound
    playsound.playsound(alarm_file)


def main():
    global COUNTER, ALARM_ON
    video_capture = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")

        # emotion detection
    while True:
        ret, frame = video_capture.read(0)
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dom_emo = result[0]['dominant_emotion']

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(faceCascade.empty())
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        # draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use putText() method for inserting text on video
        cv2.putText(frame, dom_emo, (10, 400), font, 1, (255, 0, 0), 2, cv2.LINE_AA, )

        if (dom_emo == 'sad'):
            cv2.putText(frame, 'Have some tea!', (50, 50), font, 1, (255, 0, 255), 3, cv2.LINE_AA)
            if not ALARM_ON:
                ALARM_ON = True
                t = Thread(target=sound_alarm, args=('havetea.wav',))
                # t = Thread(target=sound_alarm, args=('example.wav',))
                t.deamon = True
                t.start()

        elif (dom_emo == 'angry'):
            cv2.putText(frame, 'Wanna hear some music!', (50, 50), font, 1, (255, 0, 255), 3, cv2.LINE_AA)
            if not ALARM_ON:
                ALARM_ON = True
                t = Thread(target=sound_alarm, args=('wannahearmusic.wav',))
                # t = Thread(target=sound_alarm, args=('example.wav',))
                t.deamon = True
                t.start()

        elif (dom_emo == 'fear'):
            cv2.putText(frame, 'Have some water!', (50, 50), font, 1, (255, 0, 255), 3, cv2.LINE_AA)
            if not ALARM_ON:
                ALARM_ON = True
                t = Thread(target=sound_alarm, args=('havewater.wav',))
                # t = Thread(target=sound_alarm, args=('example.wav',))
                t.deamon = True
                t.start()

        # def commands:

        #cv2.imshow('Original video', frame)
        # get it into the correct format
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get the correct face landmarks

        face_landmarks_list = face_recognition.face_landmarks(frame)

        # get eyes
        # drowsiness detection
        for face_landmark in face_landmarks_list:
            leftEye = face_landmark['left_eye']
            rightEye = face_landmark['right_eye']
            # eye aspect ratio for left and right eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2
            # ========================converting left and right eye values in numpy arrays
            lpts = np.array(leftEye)
            rpts = np.array(rightEye)
            # ==================showing line from left of left eye and right of right eye
            cv2.polylines(frame, [lpts], True, (255, 255, 0), 1)
            cv2.polylines(frame, [rpts], True, (255, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < MINI_AER:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of times
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=sound_alarm, args=('alarm.wav',))
                        t.deamon = True
                        t.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "ALERT! You are feeling asleep!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




            elif ear < MIN_AER:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of times
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        # t = Thread(target=sound_alarm, args=('alarm.wav',))
                        t = Thread(target=sound_alarm, args=('example.wav',))
                        t.deamon = True
                        t.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "ALERT! You are feeling asleep!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("DMDS detection program.", frame)

        # ifq the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # do a bit of cleanup
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()