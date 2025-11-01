import cv2
import json
import base64
import time
import numpy as np
import socket
import sys 

# --- KONFIGURASI JARINGAN (UDP CLIENT) ---
# GANTI IP INI dengan IP dari Station Server Anda!
SERVER_IP = '127.0.0.1' 
SERVER_UDP_PORT = 8080 
IP_CAM_URL = 'http://192.168.100.90:8080/video' 
FRAME_RATE_LIMIT = 15 # Batasan FPS untuk pengiriman data UDP

# Inisialisasi UDP Socket
UDP_CLIENT_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

# --- VARIABEL GLOBAL UTAMA (STATE) ---
cap = None
GLOBAL_FRAME_RAW = None 
GLOBAL_LANE_CENTER_X = None 
GLOBAL_IS_LANE_VALID = False 
GLOBAL_OBSTACLE_DISTANCE = 120.0 

# --- VARIABEL GLOBAL UNTUK SPEED DINAMIS ---
GLOBAL_ACTUAL_SPEED = 0.0 
ACCELERATION_RATE = 0.05 

# --------------------------------------------------------------------------
#                          FUNGSI PEMBANTU (HELPERS)
# --------------------------------------------------------------------------

def encode_image(frame):
    """Mengodekan frame OpenCV ke string Base64."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return base64.b64encode(buffer).decode('utf-8')

def get_perspective_matrix(frame):
    # ... (Fungsi ini tetap sama dengan kode Anda sebelumnya) ...
    height, width = frame.shape[:2]
    src = np.float32([[width * 0.45, height * 0.65], [width * 0.55, height * 0.65], [width, height], [0, height]])
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def perspective_transform(img, M, output_size):
    # ... (Fungsi ini tetap sama dengan kode Anda sebelumnya) ...
    return cv2.warpPerspective(img, M, output_size, flags=cv2.INTER_LINEAR)

def preprocess_frame(frame_input):
    """
    Menerapkan ROI dan HLS Thresholding untuk menghasilkan BEV Mask.
    
    Langkah: ROI -> HLS Thresholding -> Morfologi.
    """
    height, width = frame_input.shape[:2]
    
    # --- 1. Definisikan Area ROI ---
    # Mulai dari 1/1.4 tinggi gambar hingga ke bawah.
    y_start = int(height / 1.4) 
    
    # Ambil ROI dari frame mentah
    roi_image = frame_input[y_start:height, 0:width]
    
    # --- 2. Konversi Warna dan Thresholding (Image Segmentation) ---
    roi_hls = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HLS) 
    
    # batasan warna
    lower_hls = np.array([26, 0, 0])
    upper_hls = np.array([255, 166, 38])
    roi_thresholded = cv2.inRange(roi_hls, lower_hls, upper_hls) 

    #bersihkan noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Iterasi Morfologi 
    roi_thresholded = cv2.erode(roi_thresholded, kernel, iterations=2)
    roi_thresholded = cv2.dilate(roi_thresholded, kernel, iterations=8)
    roi_thresholded = cv2.erode(roi_thresholded, kernel, iterations=5)
    roi_thresholded = cv2.dilate(roi_thresholded, kernel, iterations=8)
    roi_thresholded = cv2.erode(roi_thresholded, kernel, iterations=7)

    # --- 4. Gabungkan hasil mask dengan ukuran frame penuh ---
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y_start:height, 0:width] = roi_thresholded

    return full_mask

# --------------------------------------------------------------------------
#                          FUNGSI KONTROL & DETEKSI
# --------------------------------------------------------------------------

def detect_lane_lines_BEV_core(warped_mask_input):
    # ... (Salin dan tempel implementasi Centroid Tiga Jalur Anda di sini) ...
    height, width = warped_mask_input.shape[:2]
    lane_center_x = width / 2
    is_lane_valid = False

    # Definisikan tiga area
    width_third = width // 3
    left_mask = warped_mask_input[:, 0 : width_third]
    center_mask = warped_mask_input[:, width_third : 2 * width_third]
    right_mask = warped_mask_input[:, 2 * width_third : width]
    
    MIN_PIXEL_PER_SIDE = 300 
    ASSUMED_LANE_WIDTH = 300 
    
    left_x, center_x, right_x = -1, -1, -1
    
    M_left = cv2.moments(left_mask)
    if M_left["m00"] > MIN_PIXEL_PER_SIDE:
        left_x = int(M_left["m10"] / M_left["m00"])
    
    M_center = cv2.moments(center_mask)
    if M_center["m00"] > MIN_PIXEL_PER_SIDE:
        center_x = int(M_center["m10"] / M_center["m00"]) + width_third
        
    M_right = cv2.moments(right_mask)
    if M_right["m00"] > MIN_PIXEL_PER_SIDE:
        right_x = int(M_right["m10"] / M_right["m00"]) + (2 * width_third)

    if left_x != -1 and right_x != -1:
        current_width = right_x - left_x
        if current_width > (ASSUMED_LANE_WIDTH * 0.5) and current_width < (ASSUMED_LANE_WIDTH * 2.0):
            lane_center_x = (left_x + right_x) / 2
            is_lane_valid = True
    elif left_x != -1:
        lane_center_x = left_x + ASSUMED_LANE_WIDTH / 2
        is_lane_valid = True
    elif right_x != -1:
        lane_center_x = right_x - ASSUMED_LANE_WIDTH / 2
        is_lane_valid = True
    elif center_x != -1:
        lane_center_x = center_x
        is_lane_valid = True

    data_filtered = warped_mask_input.copy()
    # ... (Tambahkan kembali Visualisasi Centroid jika perlu untuk debugging lokal) ...

    return lane_center_x, is_lane_valid, data_filtered

def calculate_steering_angle():
    # ... (Salin dan tempel implementasi P-Control Anda di sini) ...
    global GLOBAL_LANE_CENTER_X, GLOBAL_IS_LANE_VALID, GLOBAL_FRAME_RAW
    if not GLOBAL_IS_LANE_VALID or GLOBAL_LANE_CENTER_X is None or GLOBAL_FRAME_RAW is None:
        return 0.0 

    width = GLOBAL_FRAME_RAW.shape[1] 
    image_center_x = width / 2
    MIN_CX_PERCENT = 0.10 * width 
    MAX_CX_PERCENT = 0.90 * width 
    steering_angle = 0.0
    
    if GLOBAL_LANE_CENTER_X > MIN_CX_PERCENT and GLOBAL_LANE_CENTER_X < MAX_CX_PERCENT:
        offset = GLOBAL_LANE_CENTER_X - image_center_x
        steering_angle = -offset * 0.1 # Koefisien P
        MAX_ANGLE = 35 
        steering_angle = np.clip(steering_angle, -MAX_ANGLE, MAX_ANGLE)

    return steering_angle

def get_target_speed(steering_angle, max_speed=50.0, min_speed=15.0, angle_threshold=15.0):
    # ... (Salin dan tempel implementasi Target Speed Anda di sini) ...
    global GLOBAL_IS_LANE_VALID
    if not GLOBAL_IS_LANE_VALID:
        return 0.0

    abs_angle = abs(steering_angle)
    
    if abs_angle < angle_threshold:
        target_speed = max_speed
    else:
        angle_range = 35.0 - angle_threshold 
        normalized_factor = (abs_angle - angle_threshold) / angle_range
        speed_reduction = (max_speed - min_speed) * normalized_factor
        target_speed = max_speed - speed_reduction
        
    return max(min_speed, target_speed)

def calculate_speed_dynamic(target_speed):
    # ... (Salin dan tempel implementasi Dynamic Speed Anda di sini) ...
    global GLOBAL_ACTUAL_SPEED, GLOBAL_IS_LANE_VALID, ACCELERATION_RATE
    if not GLOBAL_IS_LANE_VALID:
        GLOBAL_ACTUAL_SPEED = 0.0
        return 0.0

    GLOBAL_ACTUAL_SPEED += (target_speed - GLOBAL_ACTUAL_SPEED) * ACCELERATION_RATE
    return max(0.0, GLOBAL_ACTUAL_SPEED)

def get_obstacle_telemetry():
    # ... (Salin dan tempel implementasi Obstacle Anda di sini) ...
    global GLOBAL_OBSTACLE_DISTANCE, GLOBAL_IS_LANE_VALID
    
    if not GLOBAL_IS_LANE_VALID:
        return {
            "obstacleDetected": False,
            "obstacleDistance": 0.0,
            "obstaclePosition": "unknown"
        }
    
    is_detected = GLOBAL_OBSTACLE_DISTANCE < 50 
    
    return {
        "obstacleDetected": is_detected,
        "obstacleDistance": GLOBAL_OBSTACLE_DISTANCE,
        "obstaclePosition": "center"
    }

def process_frame_and_get_angle_visualization(steering_angle):
    # ... (Salin dan tempel implementasi Visualisasi Frame Anda di sini) ...
    global GLOBAL_FRAME_RAW, GLOBAL_LANE_CENTER_X, GLOBAL_IS_LANE_VALID
    
    if GLOBAL_FRAME_RAW is None:
        return None
    
    frame_processed_color = GLOBAL_FRAME_RAW.copy() 
    height, width = frame_processed_color.shape[:2]
    
    M, Minv = get_perspective_matrix(GLOBAL_FRAME_RAW)
    cv2.line(frame_processed_color, (int(width / 2), 0), (int(width / 2), height), (255, 0, 255), 2)
    
    if GLOBAL_IS_LANE_VALID and GLOBAL_LANE_CENTER_X is not None:
        point_bev = np.float32([[[GLOBAL_LANE_CENTER_X, height - 10]]]) 
        point_original = cv2.perspectiveTransform(point_bev, Minv)
        original_center_x = int(point_original[0][0][0])
        original_center_y = int(point_original[0][0][1])
        cv2.line(frame_processed_color, (original_center_x, original_center_y), 
                 (original_center_x, height), (0, 255, 0), 5) 

    status_text = "Detected" if GLOBAL_IS_LANE_VALID else "LOST"
    color = (0, 255, 0) if GLOBAL_IS_LANE_VALID else (0, 0, 255)
    cv2.putText(frame_processed_color, f"Status: {status_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame_processed_color, f"Angle: {steering_angle:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame_processed_color

# --------------------------------------------------------------------------
#                          LOGIKA PENGIRIMAN UDP
# --------------------------------------------------------------------------

def send_data_via_udp(telemetry_data):
    """Mengemas data ke JSON dan mengirimkannya via UDP."""
    # Data harus dibungkus dalam key 'data' agar sesuai dengan Station Server
    final_package = {
        "type": "robot_telemetry",
        "data": telemetry_data
    }
    
    try:
        message = json.dumps(final_package)
        UDP_CLIENT_SOCKET.sendto(message.encode('utf-8'), (SERVER_IP, SERVER_UDP_PORT))
        
        # print(f"[UDP SEND] Data dikirim. Size: {len(message)}")
    
    except socket.error as e:
        print(f"[UDP ERROR] Gagal mengirim data ke {SERVER_IP}:{SERVER_UDP_PORT}: {e}")
    except Exception as e:
        print(f"[ERROR] Kesalahan saat encoding/mengirim: {e}")


# --------------------------------------------------------------------------
#                          FUNGSI UTAMA (MAIN LOOP)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
#                          FUNGSI UTAMA (MAIN LOOP)
# --------------------------------------------------------------------------

def main_loop():
    global cap, GLOBAL_FRAME_RAW, GLOBAL_LANE_CENTER_X, GLOBAL_IS_LANE_VALID
    print("--- ROBOT PROGRAM (UDP Client) ---")

    # 1. INIT KAMERA 
    cap = cv2.VideoCapture(IP_CAM_URL, cv2.CAP_FFMPEG) 
    # ... (Pengecekan Kamera) ...

    print("SUCCESS: IP Camera berhasil dibuka. Memulai proses loop...")
    
    while True:
        start_time = time.time()
        
        # 2. BACA FRAME & UPDATE GLOBAL
        ret, frame_raw = cap.read()
        if not ret:
            print("[CAMERA WARNING] Gagal membaca frame.")
            time.sleep(0.01)
            continue
        
        GLOBAL_FRAME_RAW = frame_raw.copy() 

        # 3. PRA-PROSES CV (LOGIKA LANE DETECTION BERDASARKAN MODUL)
        # # 3.1. Pra-proses ROI, HLS Thresholding, dan Morfologi
        # # Menghasilkan mask biner dari jalur jalan.
        frame_mask = preprocess_frame(GLOBAL_FRAME_RAW) 
        
        # 3.2. Transformasi Perspektif (Bird Eye View)
        M, Minv = get_perspective_matrix(GLOBAL_FRAME_RAW)
        # Terapkan Matriks Perspektif pada hasil Masking
        frame_bev_mask = perspective_transform(frame_mask, M, GLOBAL_FRAME_RAW.shape[1::-1])

        # 4. KONTROL & LOGIKA
        GLOBAL_LANE_CENTER_X, GLOBAL_IS_LANE_VALID, _ = detect_lane_lines_BEV_core(frame_bev_mask)
        steering_angle = calculate_steering_angle()
        target_speed = get_target_speed(steering_angle)
        current_speed = calculate_speed_dynamic(target_speed)
        obstacle_data = get_obstacle_telemetry()
        frame_processed = process_frame_and_get_angle_visualization(steering_angle)
        
        # 5. PENGEMASAN DATA (TERMASUK GAMBAR BASE64)# ... (Bagian ini tetap sama) ...
        telemetry_package = {
            "steering_angle": round(steering_angle, 2), 
            "laneStatus": "Detected" if GLOBAL_IS_LANE_VALID else "Lost",
            "speed": round(current_speed, 2),
            "deviation": round(steering_angle / 10, 2), 
            "obstacleDetected": obstacle_data["obstacleDetected"],
            "obstacleDistance": obstacle_data["obstacleDistance"],
            "obstaclePosition": obstacle_data["obstaclePosition"],
            "raw_image_b64": encode_image(GLOBAL_FRAME_RAW),
            "processed_image_b64": encode_image(frame_processed) if frame_processed is not None else ""
        }
        
        # 6. KIRIM VIA UDP
        send_data_via_udp(telemetry_package)
        
        # 7. PENGATURAN FPS
        elapsed_time = time.time() - start_time
        delay = (1.0 / FRAME_RATE_LIMIT) - elapsed_time
        if delay > 0:
            time.sleep(delay)
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[STOP] Program dihentikan oleh pengguna (Ctrl+C).")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        print("[CLEANUP] Kamera dan sumber daya ditutup.")
