#Face Recognation 
import cv2
import os
import time

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = int(input("Berapa orang yang akan dideteksi? "))
max_images = 30 # max images to be taken for each person

for person_id in range(1, person_id + 1):
    print(f"[INFO] Get ready to capture data for person ID {person_id}. Position your face in front of the camera.")
    time.sleep(5) # Give 3 seconds to get ready
    print("[INFO] Starting image capture...")

    count = 0
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

        

        if cv2.waitKey(1) == ord('q') or count >= max_images:
            break

    print(f"[INFO] Selesai mengambil {count} gambar untuk orang dengan ID {person_id}.")
    if person_id > 1:
        print("Tekan tombol apa saja untuk melanjutkan ke orang berikutnya atau 'q' untuk keluar.")
        if cv2.waitKey(0) == ord('q'):
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
import os 
# Ambil semua file gambar di folder dataset
image_files = [f for f in os.listdir("dataset") if f.endswith(".jpg")]

# Ambil ID unik dari nama file (misal: Person-1-5.jpg -> ID = 1)
ids = set()
for filename in image_files:
    id = int(filename.split("-")[1])
    ids.add(id)

ids = sorted(list(ids))  # urutkan ID
names = ["None"]         # index 0 tetap "None"

for id in ids:
    name = input(f"Masukkan nama untuk ID {id}: ")
    names.append(name)

print("List nama sesuai ID:", names)


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
            if id < len(names):
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
