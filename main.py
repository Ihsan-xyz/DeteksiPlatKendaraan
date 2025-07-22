# main.py

import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from sort.sort import Sort
import numpy as np
import os

from util import get_car, read_license_plate, log_detection

# --- KONFIGURASI ---
CAMERA_INDEX = 0  
MODEL_VEHICLE_PATH = 'yolov8n.pt'
MODEL_PLATE_PATH = 'license_plate_detector.pt'
LOG_FILE_PATH = 'detection_log.csv'

# Kelas ID untuk kendaraan di model COCO (2: mobil, 3: motor, 5: bus, 7: truk)
VEHICLE_CLASSES = [2, 3, 5, 7]

# --- OPTIMASI PERFORMA ---
TARGET_WIDTH = 480
FRAME_SKIP = 4

# --- PENGATURAN MODE HEMAT DAYA ---
POWER_SAVE_INTERVAL = 1.0

def main():
    """Fungsi utama untuk menjalankan ALPR secara real-time dengan optimasi."""
    
    # --- INISIALISASI ---
    print("Initializing models...")
    if not os.path.exists(MODEL_VEHICLE_PATH) or not os.path.exists(MODEL_PLATE_PATH):
        print(f"Error: Model file not found. Make sure '{MODEL_VEHICLE_PATH}' and '{MODEL_PLATE_PATH}' exist.")
        return

    coco_model = YOLO(MODEL_VEHICLE_PATH)
    plate_detector = YOLO(MODEL_PLATE_PATH)
    mot_tracker = Sort()

    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    # Variabel manajemen status
    power_save_mode = False
    last_scan_time = 0
    frame_count = 0
    last_logged_plate_for_id = {}
    last_tracked_vehicles = []

    print("ALPR system is running. Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from camera. Exiting.")
                break

            # OPTIMASI 1: Ubah ukuran frame untuk mempercepat pemrosesan
            height, width, _ = frame.shape
            scale = TARGET_WIDTH / width
            target_height = int(height * scale)
            frame = cv2.resize(frame, (TARGET_WIDTH, target_height))

            frame_count += 1
            # OPTIMASI 2: Lewati frame untuk mengurangi beban CPU
            if frame_count % (FRAME_SKIP + 1) != 0 and not power_save_mode:
                # Tetap gambar kotak dari deteksi terakhir agar visual tetap update
                perform_processing = False
            else:
                perform_processing = True

            if power_save_mode:
                if time.time() - last_scan_time >= POWER_SAVE_INTERVAL:
                    print("Scanning for activity...")
                    last_scan_time = time.time()
                    perform_processing = True
                else:
                    perform_processing = False

            if perform_processing:
                # DETEKSI KENDARAAN
                vehicle_detections = coco_model(frame, verbose=False)[0]
                vehicle_detections_ = []
                for detection in vehicle_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in VEHICLE_CLASSES and score > 0.5:
                        vehicle_detections_.append([x1, y1, x2, y2, score])

                # Manajemen Mode Hemat Daya
                if not vehicle_detections_:
                    if not power_save_mode:
                        print("No vehicles detected. Entering power-save mode.")
                        power_save_mode = True
                    last_tracked_vehicles = []
                else:
                    if power_save_mode:
                        print("Vehicle detected. Resuming real-time tracking.")
                        power_save_mode = False

                if vehicle_detections_:
                    last_tracked_vehicles = mot_tracker.update(np.asarray(vehicle_detections_))
                    current_vehicle_ids = {int(v[4]) for v in last_tracked_vehicles}
                    for car_id in list(last_logged_plate_for_id.keys()):
                        if car_id not in current_vehicle_ids:
                            del last_logged_plate_for_id[car_id]
                    
                    # OPTIMASI 3: Deteksi plat hanya di area kendaraan (ROI)
                    for vehicle in last_tracked_vehicles:
                        vx1, vy1, vx2, vy2, car_id = map(int, vehicle)
                        
                        # Potong area kendaraan dari frame
                        vehicle_crop = frame[vy1:vy2, vx1:vx2]
                        if vehicle_crop.size == 0: continue

                        # Jalankan deteksi plat pada potongan kecil
                        license_plates_in_vehicle = plate_detector(vehicle_crop, verbose=False)[0]

                        for plate in license_plates_in_vehicle.boxes.data.tolist():
                            px, py, px2, py2, p_score, _ = plate
                            
                            # Koordinat plat relatif terhadap frame utama
                            full_px1, full_py1 = vx1 + int(px), vy1 + int(py)
                            full_px2, full_py2 = vx1 + int(px2), vy1 + int(py2)
                            
                            # Gambar kotak plat di frame utama
                            cv2.rectangle(frame, (full_px1, full_py1), (full_px2, full_py2), (0, 0, 255), 2)
                            
                            plate_crop_img = vehicle_crop[int(py):int(py2), int(px):int(px2)]
                            if plate_crop_img.size == 0: continue

                            plate_text, text_score = read_license_plate(plate_crop_img)

                            if plate_text is not None:
                                if last_logged_plate_for_id.get(car_id) != plate_text:
                                    log_detection(LOG_FILE_PATH, plate_text, datetime.now())
                                    last_logged_plate_for_id[car_id] = plate_text

            # --- VISUALISASI ---
            status_text = "Mode: Memproses"
            if power_save_mode: status_text = "Mode: Hemat Daya"
            elif not perform_processing: status_text = "Mode: Standby"

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Gambar kotak kendaraan dari deteksi terakhir yang valid
            for vehicle in last_tracked_vehicles:
                x1, y1, x2, y2, vehicle_id = map(int, vehicle)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Real-time ALPR (Optimized)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()