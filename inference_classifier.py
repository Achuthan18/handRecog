import cv2
import mediapipe as mp
import pickle
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2




mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

data_aux=[]
labels_dict={0:'1',1:'2',2:'3'}

model_dict=pickle.load(open('./model.p','rb'))
model=model_dict['model']



hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range (len(hand_landmarks.landmark)):
               x=hand_landmarks.landmark[i].x
               y=hand_landmarks.landmark[i].y
               data_aux.append(x)
               data_aux.append(y)
        data_aux=data_aux[0:42]

        prediction=model.predict([np.asarray(data_aux)])
        predicted_char=labels_dict[int(prediction[0])]
        print(predicted_char)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()    