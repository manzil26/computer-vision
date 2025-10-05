import cv2
import os
import time

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

total_person = 3
max_images = 30

for person_id in range(1, total_person + 1):
    print(f"[INFO] Siap-siap untuk mengambil data orang dengan ID {person_id}. Posisikan wajah Anda di depan kamera.")
    time.sleep(3) # Beri waktu 3 detik untuk bersiap
    print("[INFO] Pengambilan gambar dimulai...")
    
    count = 0
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = gray[y:y + h, x:x + w]
            
            # Hanya simpan gambar jika wajah terdeteksi dan ukurannya valid
            if face_roi.size > 0:
                cv2.imwrite(f"{dataset_path}Person-{person_id}-{count}.jpg", face_roi)
                count += 1
            
        cv2.imshow("Ambil Data", frame)

        if cv2.waitKey(1) == ord('q') or count >= max_images:
            break
            
    print(f"[INFO] Selesai mengambil {count} gambar untuk orang dengan ID {person_id}.")
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
    imagePath = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    ids = np.array([], dtype="int")
    
    for i in imagePath:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        filename = os.path.basename(i)
        
        # Ekstrak ID dari nama file (contoh: Person-1-0.jpg -> id = 1)
        try:
            id = int(filename.split("-")[1])
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                faces.append(img)
                ids = np.append(ids, id)
        except (ValueError, IndexError):
            print(f"Melewatkan file dengan nama tidak valid: {filename}")
            continue

    return faces, ids

if not checkDataset():
    print("Dataset tidak ditemukan atau kosong. Silakan jalankan skrip pengambilan data terlebih dahulu.")
else:
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    print("Memulai pelatihan wajah...")
    
    faces, ids = organizeDataset()
    if len(faces) > 0 and len(ids) > 0:
        recognizer.train(faces, ids)
        recognizer.write('face-model.yml')
        print("Pelatihan selesai! Model berhasil disimpan sebagai 'face-model.yml'.")
    else:
        print("Gagal melatih model. Pastikan dataset memiliki gambar yang valid.")

import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
try:
    recognizer.read("face-model.yml")
except cv2.error:
    print("Error: Model face-model.yml tidak ditemukan. Silakan latih model terlebih dahulu.")
    exit()

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

id_names = ['None', 'Manzil', 'Fiqih', 'Shafly'] # Pastikan urutan ini benar: ID 1 = Manzil, ID 2 = Fiqih, dll.
confidence_threshold = 70  # Nilai ambang batas: semakin kecil semakin akurat

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < confidence_threshold:
            name = id_names[id]
            confidence_str = f"Confidence: {round(100 - confidence)}%"
        else:
            name = "unknown"
            confidence_str = f"Confidence: {round(100 - confidence)}%"

        cv2.putText(frame, str(name), (x + 5, y - 5), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, confidence_str, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow("Deteksi Wajah", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
