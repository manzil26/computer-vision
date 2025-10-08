# image_recognation.py
# Face Recognition + tampilkan umur & status + wajah tak dikenal (merah)
import cv2
import os
import time
import numpy as np
from datetime import datetime

# === Inisialisasi ===
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

# === Pengambilan Dataset ===
total_person = int(input("Berapa orang yang akan dideteksi? "))
for person_id in range(1, total_person + 1):
    print(f"[INFO] Siapkan wajah untuk ID {person_id}")
    time.sleep(2)
    count = 0
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_name = os.path.join(dataset_path, f"Person-{person_id}-{count}.jpg")
            cv2.imwrite(file_name, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow("Capture (press q to stop)", frame)
        if cv2.waitKey(1) == ord('q') or count >= 50:
            break
    print(f"[INFO] Selesai mengambil {count} gambar untuk ID {person_id}")

cap.release()
cv2.destroyAllWindows()

# === Fungsi Pendukung ===
def checkDataset(path=dataset_path):
    return any(f.endswith(".jpg") for f in os.listdir(path))

def organizeDataset(path=dataset_path):
    faces, ids = [], []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            try:
                id = int(file.split("-")[1])
                img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                ids.append(id)
            except:
                pass
    return faces, np.array(ids, dtype=np.int32)

# === Training Model ===
if not checkDataset():
    print("Dataset tidak ditemukan. Buat dataset terlebih dahulu.")
else:
    print("[INFO] Training model wajah...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = organizeDataset()
    if len(faces) == 0:
        print("[ERROR] Tidak ada wajah untuk dilatih.")
    else:
        recognizer.train(faces, ids)
        recognizer.write("face-model.yml")
        print("[INFO] Training selesai dan model tersimpan sebagai face-model.yml")

# ...existing code...
# === Input Nama & Tahun Lahir ===
names = ["None"]
birth_years = [None]

for id in range(1, total_person + 1):
    name = input(f"Masukkan nama untuk ID {id}: ")
    while True:
        by = input(f"Masukkan tahun lahir untuk ID {id} (contoh: 1998): ")
        try:
            by_int = int(by)
            if 1900 <= by_int <= datetime.now().year:
                break
            else:
                print("Tahun lahir tidak valid.")
        except:
            print("Masukkan angka yang valid.")
    names.append(name)
    birth_years.append(by_int)
# ...existing code...

# === Pengenalan Wajah Real-Time ===
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists("face-model.yml"):
    recognizer.read("face-model.yml")
else:
    print("[ERROR] Model tidak ditemukan.")
    exit()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

print("[INFO] Sistem pengenalan wajah dimulai. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        id_pred, confidence = recognizer.predict(face_roi)

        display_name = "Tidak Terdeteksi"
        display_conf = f"{round(100 - confidence)}%"
        display_age = ""
        display_status = ""
        color = (0, 0, 255)  # default merah untuk tidak terdeteksi

        # Jika confidence tinggi → wajah dikenali
        if confidence < 100:
            if 0 <= id_pred < len(names):
                display_name = f"Nama: {names[id_pred]}"
                birth_year = birth_years[id_pred]
                if birth_year:
                    age = datetime.now().year - birth_year
                    display_age = f"Umur: {age}"
                    display_status = "Status: Dewasa" if age >= 18 else "Status: Anak"
                color = (0, 255, 0)  # hijau untuk terdeteksi
            else:
                display_name = "Tidak Terdeteksi"

        # Kotak wajah (warna sesuai hasil)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Semua keterangan di atas wajah
        y_offset = y - 10
        line_height = 25
        info_lines = [display_name]
        if display_age:
            info_lines.append(display_age)
        if display_status:
            info_lines.append(display_status)
        info_lines.append(f"Confidence: {display_conf}")

        for i, text in enumerate(info_lines[::-1]):
            y_pos = y_offset - i * line_height
            cv2.putText(frame, text, (x, max(30, y_pos)), font, 0.6, color, 2)

    cv2.imshow("Pengenalan Wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program dihentikan.")
