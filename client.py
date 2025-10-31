import asyncio
import websockets
import cv2
import json
import base64
import time
import numpy as np

# --- KONFIGURASI JARINGAN & KAMERA ---
IP_CAM_URL = 'http://10.7.101.210:8080/video' 
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8081
FRAME_RATE_LIMIT = 30 

cap = None 
cap_lock = asyncio.Lock() 

GLOBAL_FRAME_RAW = None 
GLOBAL_LANE_CENTER_X = None # Posisi X tengah jalur di BEV
GLOBAL_IS_LANE_VALID = False # Status deteksi jalur (KRITIS!)
GLOBAL_OBSTACLE_DISTANCE = 120.0 # Contoh nilai dummy

GLOBAL_ACTUAL_SPEED = 0.0 # Kecepatan yang diintegrasikan dan dikirim
ACCELERATION_RATE = 0.05 # Faktor akselerasi (kecepatan perubahan speed)


def create_telemetry_message_new(angle, is_lane_valid, current_speed, obstacle_data):
    lane_status = "Detected" if is_lane_valid else "Lost"
    
    return json.dumps({
        "type": "telemetry",
        "data": {
            "steering_angle": angle,  
            "laneStatus": lane_status,
            "speed": current_speed,
            "deviation": angle / 10, 
            "obstacleDetected": obstacle_data["obstacleDetected"],
            "obstacleDistance": obstacle_data["obstacleDistance"],
            "obstaclePosition": obstacle_data["obstaclePosition"]
        }
    })

def encode_image(frame, img_type, width, height):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    b64_data = base64.b64encode(buffer).decode('utf-8')
    return json.dumps({
        "type": f"image_{img_type}",
        "data": b64_data,
        "width": width,
        "height": height
    })

def get_perspective_matrix(frame):
    height, width = frame.shape[:2]
    # !!! NILAI INI ADALAH DUMMY DAN HARUS DISESUAIKAN !!!
    src = np.float32([[width * 0.45, height * 0.65], [width * 0.55, height * 0.65], [width, height], [0, height]])
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def perspective_transform(img, M, output_size):
    return cv2.warpPerspective(img, M, output_size, flags=cv2.INTER_LINEAR)

# LANE DETECTION

def detect_lane_lines_BEV_core(warped_mask_input):
    height, width = warped_mask_input.shape[:2]
    lane_center_x = width / 2
    left_avg_x = -1 
    right_avg_x = -1 
    is_lane_valid = False
    data_filtered = warped_mask_input.copy() 

    # centroid
    left_half_mask = data_filtered[:, 0:width//2]
    right_half_mask = data_filtered[:, width//2:width]
    MIN_PIXEL_PER_SIDE = 300 
    ASSUMED_LANE_WIDTH = 300 

    M_left = cv2.moments(left_half_mask)
    if M_left["m00"] > MIN_PIXEL_PER_SIDE:
        left_avg_x = int(M_left["m10"] / M_left["m00"])
        
    M_right = cv2.moments(right_half_mask)
    if M_right["m00"] > MIN_PIXEL_PER_SIDE:
        right_avg_x = int(M_right["m10"] / M_right["m00"]) + (width//2) 

    if left_avg_x != -1 and right_avg_x != -1:
        current_width = right_avg_x - left_avg_x
        if current_width > (ASSUMED_LANE_WIDTH * 0.5) and current_width < (ASSUMED_LANE_WIDTH * 1.5):
            lane_center_x = (left_avg_x + right_avg_x) / 2
            is_lane_valid = True
    elif left_avg_x != -1:
        lane_center_x = left_avg_x + ASSUMED_LANE_WIDTH / 2
        is_lane_valid = True
    elif right_avg_x != -1:
        lane_center_x = right_avg_x - ASSUMED_LANE_WIDTH / 2
        is_lane_valid = True

    return lane_center_x, is_lane_valid, data_filtered

def detect_lane_lines_BEV_core(warped_mask_input):
    height, width = warped_mask_input.shape[:2]
    lane_center_x = width / 2
    is_lane_valid = False
    
    # Definisikan tiga area
    width_third = width // 3
    
    # kiri
    left_mask = warped_mask_input[:, 0 : width_third]
    
    # tengah
    center_mask = warped_mask_input[:, width_third : 2 * width_third]
    
    # kanan
    right_mask = warped_mask_input[:, 2 * width_third : width]

    MIN_PIXEL_PER_SIDE = 300  
    ASSUMED_LANE_WIDTH = 300 

    left_x, center_x, right_x = -1, -1, -1

    # Deteksi Garis Kiri
    M_left = cv2.moments(left_mask)
    if M_left["m00"] > MIN_PIXEL_PER_SIDE:
        left_x = int(M_left["m10"] / M_left["m00"])
        
    # Deteksi Garis Tengah
    M_center = cv2.moments(center_mask)
    if M_center["m00"] > MIN_PIXEL_PER_SIDE:
        center_x = int(M_center["m10"] / M_center["m00"]) + width_third
    
    # Deteksi Garis Kanan
    M_right = cv2.moments(right_mask)
    if M_right["m00"] > MIN_PIXEL_PER_SIDE:
        right_x = int(M_right["m10"] / M_right["m00"]) + (2 * width_third)

    # --- Logika Penentuan Center Jalur (Averaging) ---

    # 1. Jika Kiri dan Kanan ditemukan (Paling Prioritas)
    if left_x != -1 and right_x != -1:
        current_width = right_x - left_x
        if current_width > (ASSUMED_LANE_WIDTH * 0.5) and current_width < (ASSUMED_LANE_WIDTH * 2.0):
            lane_center_x = (left_x + right_x) / 2
            is_lane_valid = True
            
    # 2. Jika hanya satu garis terdeteksi (Fallback menggunakan asumsi lebar)
    elif left_x != -1:
        lane_center_x = left_x + ASSUMED_LANE_WIDTH / 2
        is_lane_valid = True
    elif right_x != -1:
        lane_center_x = right_x - ASSUMED_LANE_WIDTH / 2
        is_lane_valid = True
        
    # 3. Jika hanya Tengah yang terdeteksi (Fallback terakhir)
    elif center_x != -1:
        lane_center_x = center_x
        is_lane_valid = True

    # Visualisasi Centroid (Untuk debugging)
    data_filtered = warped_mask_input.copy()
    if left_x != -1:
        cv2.circle(data_filtered, (left_x, height-10), 10, (255, 0, 0), -1)
    if center_x != -1:
        cv2.circle(data_filtered, (center_x, height-10), 10, (255, 255, 255), -1)
    if right_x != -1:
        cv2.circle(data_filtered, (right_x, height-10), 10, (0, 0, 255), -1)
    
    if is_lane_valid:
        cv2.circle(data_filtered, (int(lane_center_x), height-50), 15, (0, 255, 0), -1)

    return lane_center_x, is_lane_valid, data_filtered
# 2. STEERING ANGLE 

def calculate_steering_angle():
    """Menghitung sudut kemudi (P-Control)."""
    global GLOBAL_LANE_CENTER_X, GLOBAL_IS_LANE_VALID

    # GUARD CLAUSE: Jika Lost, kembalikan 0.0
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

# --- 3. SPEED ---

def get_target_speed(steering_angle, max_speed=50.0, min_speed=15.0, angle_threshold=15.0):
    """Menghitung TARGET SPEED IDEAL berdasarkan sudut kemudi."""
    global GLOBAL_IS_LANE_VALID
    
    # Target Speed akan 0 jika Lost
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
    """Mengintegrasikan GLOBAL_ACTUAL_SPEED menuju target_speed secara bertahap."""
    global GLOBAL_ACTUAL_SPEED, GLOBAL_IS_LANE_VALID, ACCELERATION_RATE
    
    # GUARD CLAUSE: Jika Lost (target_speed = 0), langsung atur ke 0
    if not GLOBAL_IS_LANE_VALID:
        GLOBAL_ACTUAL_SPEED = 0.0
        return 0.0
        
    # Gerak akselerasi/deselerasi yang lebih halus
    GLOBAL_ACTUAL_SPEED += (target_speed - GLOBAL_ACTUAL_SPEED) * ACCELERATION_RATE
    
    return max(0.0, GLOBAL_ACTUAL_SPEED)

# --- 4. JARAK/OBSTACLE ---

def get_obstacle_telemetry():
    """Mengembalikan data obstacle yang diasumsikan."""
    global GLOBAL_OBSTACLE_DISTANCE, GLOBAL_IS_LANE_VALID

    # GUARD CLAUSE: Jika Lost, kembalikan Jarak 0
    if not GLOBAL_IS_LANE_VALID:
        return {
            "obstacleDetected": False,
            "obstacleDistance": 0.0,
            "obstaclePosition": "unknown"
        }
        
    # Logika dummy
    is_detected = GLOBAL_OBSTACLE_DISTANCE < 50 
    
    return {
        "obstacleDetected": is_detected,
        "obstacleDistance": GLOBAL_OBSTACLE_DISTANCE,
        "obstaclePosition": "center"
    }

# --- 5. VISUALISASI FRAME ---

def process_frame_and_get_angle_visualization(steering_angle):
    """Menambahkan visualisasi ke GLOBAL_FRAME_RAW dan mengembalikan frame yang diproses."""
    global GLOBAL_FRAME_RAW, GLOBAL_LANE_CENTER_X, GLOBAL_IS_LANE_VALID
    
    if GLOBAL_FRAME_RAW is None:
        return None
        
    frame_processed_color = GLOBAL_FRAME_RAW.copy() 
    height, width = frame_processed_color.shape[:2]
    
    M, Minv = get_perspective_matrix(GLOBAL_FRAME_RAW)

    # Visualisasi Garis Tengah Frame Asli (Magenta)
    cv2.line(frame_processed_color, (int(width / 2), 0), (int(width / 2), height), (255, 0, 255), 2)
    
    # Visualisasi Proyeksi Jalur (Hanya jika terdeteksi)
    if GLOBAL_IS_LANE_VALID and GLOBAL_LANE_CENTER_X is not None:
        point_bev = np.float32([[[GLOBAL_LANE_CENTER_X, height - 10]]]) 
        point_original = cv2.perspectiveTransform(point_bev, Minv)
        original_center_x = int(point_original[0][0][0])
        original_center_y = int(point_original[0][0][1])
        cv2.line(frame_processed_color, (original_center_x, original_center_y), 
                 (original_center_x, height), (0, 255, 0), 5) 
                 
    # Tambahkan teks sudut kemudi & status
    status_text = "Detected" if GLOBAL_IS_LANE_VALID else "LOST"
    color = (0, 255, 0) if GLOBAL_IS_LANE_VALID else (0, 0, 255)
    cv2.putText(frame_processed_color, f"Status: {status_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame_processed_color, f"Angle: {steering_angle:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame_processed_color

# main loop 

async def serve_bs_connection(websocket):
    """Fungsi handler yang dieksekusi saat Base Station terhubung."""
    global GLOBAL_FRAME_RAW 
    print(f"\n[CONNECTION] Base Station berhasil terhubung dari {websocket.remote_address}")
    
    if cap is None or not cap.isOpened():
        error_msg = "FATAL: Kamera tidak terbuka. Hubungan ke BS ditutup."
        await websocket.send(json.dumps({"type": "error", "message": error_msg}))
        return

    try:
        while True:
            start_time = time.time()
            
            # 1. I/O: BACA FRAME KE GLOBAL DENGAN LOCK
            async with cap_lock:
                ret, frame_raw = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue
                GLOBAL_FRAME_RAW = frame_raw.copy() 
                
            height, width = GLOBAL_FRAME_RAW.shape[:2]
            
            # 2. KONTROL: JALANKAN SEMUA FUNGSI TERPISAH SECARA BERURUTAN
            
            # A. Deteksi Jalur (Mengubah Global State)
            detect_lane_lines_BEV_core() 
            
            # B. Hitung Sudut Kemudi
            steering_angle = calculate_steering_angle()
            
            # C. Hitung Kecepatan
            target_speed = get_target_speed(steering_angle)
            current_speed = calculate_speed_dynamic(target_speed) # Kecepatan yang berubah bertahap
            
            # D. Ambil Data Jarak
            obstacle_data = get_obstacle_telemetry()
            
            # 3. VISUALISASI
            frame_processed = process_frame_and_get_angle_visualization(steering_angle)
            if frame_processed is None:
                 await asyncio.sleep(0.01)
                 continue

            # 4. ENCODING & PENGIRIMAN
            msg_raw = encode_image(GLOBAL_FRAME_RAW, "raw", width, height)
            await websocket.send(msg_raw)
            
            msg_proc = encode_image(frame_processed, "processed", width, height)
            await websocket.send(msg_proc)

            msg_telemetry = create_telemetry_message_new(
                steering_angle, 
                GLOBAL_IS_LANE_VALID, 
                current_speed, 
                obstacle_data
            )
            await websocket.send(msg_telemetry)
            
            # 5. Pengaturan FPS
            elapsed_time = time.time() - start_time
            delay = (1.0 / FRAME_RATE_LIMIT) - elapsed_time
            if delay > 0:
                await asyncio.sleep(delay)
                
    except websockets.exceptions.ConnectionClosed:
        print(f"[DISCONNECT] Base Station terputus: {websocket.remote_address}")
    except Exception as e:
        print(f"[ERROR] Kesalahan saat beroperasi: {e}")

async def main():
    """Fungsi inisialisasi, menangani Kamera dan Server WebSocket."""
    global cap 
    
    print("--- ROBOT CLIENT (WEBSOCKET SERVER) ---")
    
    # 1. INIT KAMERA 
    cap = cv2.VideoCapture(IP_CAM_URL, cv2.CAP_FFMPEG) 
    
    if not cap.isOpened():
        print(f"FATAL ERROR: Gagal terhubung ke IP Camera di {IP_CAM_URL}.")
        return 

    print("SUCCESS: IP Camera berhasil dibuka.")
    
    # 2. MULAI SERVER WEBSOCKET
    print(f"INFO: Mencoba memulai server di ws://{SERVER_HOST}:{SERVER_PORT}")
    
    async with websockets.serve(serve_bs_connection, host=SERVER_HOST, port=SERVER_PORT):
        print("INFO: Server WebSocket berhasil dimulai.")
        await asyncio.Future() # Tahan server agar tetap berjalan

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] Server dihentikan oleh pengguna (Ctrl+C).")
    except OSError as e:
        if "address already in use" in str(e):
            print(f"ERROR: Port {SERVER_PORT} sudah digunakan.")
        else:
            raise
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows() 
        print("[CLEANUP] Kamera dan sumber daya ditutup.")
