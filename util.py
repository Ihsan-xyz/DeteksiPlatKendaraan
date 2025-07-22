# util.py

import cv2
import pandas as pd
import os
from datetime import datetime
import numpy as np
import re # Import library untuk Regular Expression

# Asumsi kita menggunakan EasyOCR sebagai library OCR yang lebih andal
# Anda perlu menginstalnya: pip install easyocr
import easyocr

# --- INISIALISASI OCR READER ---
# Inisialisasi reader secara normal.
reader = easyocr.Reader(['id', 'en'], gpu=True)

# Definisikan karakter yang valid untuk plat nomor.
# Ini akan digunakan nanti di dalam fungsi read_license_plate.
CHAR_LIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def preprocess_for_ocr(image):
    """
    Menerapkan pra-pemrosesan agresif pada gambar untuk memaksimalkan akurasi OCR.
    1. Konversi ke Grayscale: Fokus pada bentuk & kontras.
    2. Denoising: Menghilangkan noise kecil yang bisa mengganggu OCR.
    3. Adaptive Thresholding: Mengubah gambar menjadi hitam putih murni, sangat efektif
       untuk menonjolkan teks dalam kondisi cahaya yang tidak seragam.
    """
    # 1. Konversi ke Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Terapkan Denoising (Bilateral Filter)
    # denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. Terapkan Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def read_license_plate(license_plate_crop):
    """
    Membaca teks dari gambar pelat nomor yang sudah di-crop dan dioptimalkan.
    Fungsi ini sekarang menggabungkan hasil deteksi dan memvalidasinya.
    """
    processed_plate = preprocess_for_ocr(license_plate_crop)

    # --- PERBAIKAN UTAMA DI SINI ---
    # Tambahkan parameter 'allowlist' langsung ke dalam pemanggilan readtext.
    detections = reader.readtext(processed_plate, 
                                 detail=1, 
                                 batch_size=4, 
                                 width_ths=0.8,
                                 allowlist=CHAR_LIST) # <--- Parameter ditambahkan di sini

    if not detections:
        return None, None

    # Gabungkan semua teks yang terdeteksi
    full_text = ""
    total_score = 0
    
    for _, text, score in detections:
        clean_text = text.upper().replace(' ', '').replace('.', '').replace('-', '')
        full_text += clean_text
        total_score += score

    # Validasi format dasar plat nomor Indonesia menggunakan Regex
    pattern = re.compile(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$')
    
    if pattern.match(full_text) and len(full_text) >= 5:
        avg_score = total_score / len(detections)
        if avg_score > 0.4:
            return full_text, avg_score
    
    return None, None


def get_car(license_plate_bbox, track_ids):
    """
    Mencocokkan bounding box pelat nomor dengan bounding box mobil yang dilacak.
    (Fungsi ini tidak diubah)
    """
    x1, y1, x2, y2 = license_plate_bbox
    for track_id in track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track_id
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1

def log_detection(log_filepath, plate_number, timestamp):
    """
    Mencatat deteksi baru ke dalam file CSV.
    (Fungsi ini tidak diubah)
    """
    formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    log_data = {'timestamp': [formatted_time], 'license_plate': [plate_number]}
    df = pd.DataFrame(log_data)

    if not os.path.exists(log_filepath):
        df.to_csv(log_filepath, index=False)
        print(f"Log file created at {log_filepath}")
    else:
        df.to_csv(log_filepath, mode='a', header=False, index=False)

    output_string = f"Plat Nomor: {plate_number}, Waktu: {formatted_time}"
    print(output_string)