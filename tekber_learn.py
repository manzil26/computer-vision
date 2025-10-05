# # pip install opencv-python #main modul 
# # pip install opencv-contrib-python # main modul + extra modul ( yg dipakai Contrib )
# # conda install -c conda-forge opencv

# #================================================
# #image detection biasa tanpa box 
# import cv2 
# #akses web cam 
# #id dari camera misalnya jika di laptop ada 2 camera maka bisa jadi 0 1 
# camera = cv2.VideoCapture(0)
# #looping video tak hingga gunananya untuk mendapatkan video, jadi vdieo meruoakan farame yang banyak jadi harus di looping 
# while True: 
#   _, frame = camera.read() #mendapatkan frame (mengolah gelombang analog jadi camera )
#   cv2.imshow("Camera", frame)
#   if cv2.waitKey(1) == ord('q'): #shorcut untuk mematikan camera 
#     break
# camera.release()
# cv2.destroyAllWindows()

# #================================================


# #image detection dengan box

# import cv2

# cascade_path = "haarcascade_frontalface_default.xml"
# clf = cv2.CascadeClassifier(cascade_path) #load classifier untuk mendeteksi wajah 

# camera = cv2.VideoCapture(0) #membuka web cam 

# while True: # ambil frame secara terus menerus
#     _, frame = camera.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #konversi warna ke gray 
#     faces = clf.detectMultiScale( 
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         flags=cv2.CASCADE_SCALE_IMAGE,
#         minSize=(30, 30)
#     )

#     for (x, y, width, height) in faces:
#         cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)

#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()


# #================================================
# #hans detection and skeleton

# #pip install mediapipe 
# import cv2
# import mediapipe as mp

# cap = cv2.VideoCapture(0)
# hands = mp.solutions.hands.Hands()
# draws = mp.solutions.drawing_utils

# while True:
#     _, frame = cap.read()

#     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     hand_obj = hands.process(frameRGB)

#     if hand_obj.multi_hand_landmarks:
#         for hand_landmarks in hand_obj.multi_hand_landmarks:
#             draws.draw_landmarks(frame, hand_landmarks,
#                                  mp.solutions.hands.HAND_CONNECTIONS)

#     cv2.imshow("Hand Skeleton", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


#================================================
#Face Recognation 
import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = 1 # id for person that we will detect
count = 0 # count for image name id
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count+=1
        cv2.imwrite(dataset_path+"Person-"+str(person_id)
                    +"-"+str(count)+".jpg", gray[y:y+h, x:x+w])

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 30: # stop when 30 photos have been taken
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os

def checkDataset(directory="dataset/"):
    if os.path.exists(directory) and len(os.listdir(directory)) != 0:
        return True
    return False

def organizeDataset(path="dataset/"):
    imagePath = [os.path.join(path, p) for p in os.listdir(path)]
    faces = []
    ids = np.array([], dtype="int")
    for i in imagePath:
        img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
        filename = os.path.basename(i)   
        id = int(filename.split("-")[1]) 
        face = faceCascade.detectMultiScale(img)
        for (x, y, w, h) in face:
            faces.append(img[y:y+h, x:x+w])
            ids = np.append(ids, id)
    return faces, ids

if not checkDataset():
    print("Dataset not found")
else:
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # train faces
    print("Training faces...")
    faces, ids = organizeDataset()
    recognizer.train(faces, ids)
    print("Training finished!")

    # save model
    recognizer.write('face-model.yml')
    print("Model saved as 'Face-model.yml'")

import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml") # face model from face_training.py
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ['None', 'Manzil']
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 100:
            id = names[id]
        else:
            id = "unknown"
        confidence = "{}%".format(round(100-confidence))

        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255,0,0), 1)
        cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1,
                    (255,255,0), 1)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
