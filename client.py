import asyncio
import websockets
import cv2
import json
import base64
import time
import numpy as np

# --- KONFIGURASI JARINGAN & KAMERA ---
IP_CAM_URL = 'http://192.168.1.9:8080/video' 
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8081
FRAME_RATE_LIMIT = 30 

# Variabel Global untuk Kamera
# Dibuat di luar agar bisa diakses dan diinisialisasi sekali di main()
cap = None 
cap_lock = asyncio.Lock() # Lock untuk mencegah akses kamera dari thread yang berbeda

# --- FUNGSI PEMBANTU (Tidak diubah, fungsinya sudah benar) ---

def create_telemetry_message(angle, obstacle_status):
    """Membuat pesan telemetri JSON untuk Base Station."""
    return json.dumps({
        "type": "telemetry",
        "data": {
            "steering_angle": angle,  
            "laneStatus": "Detected", 
            "speed": 50.0,
            "deviation": angle / 10, 
            "obstacleDetected": obstacle_status,
            "obstacleDistance": 120,
            "obstaclePosition": "center"
        }
    })

def encode_image(frame, img_type, width, height):
    """Mengodekan frame OpenCV ke Base64 dalam format JSON yang diharapkan BS."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    b64_data = base64.b64encode(buffer).decode('utf-8')
    return json.dumps({
        "type": f"image_{img_type}",
        "data": b64_data,
        "width": width,
        "height": height
    })

def process_frame_and_get_angle(frame_raw):
    """Simulasi pemrosesan citra dan perhitungan sudut."""
    hsv = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])
    frame_processed_mono = cv2.inRange(hsv, lower_bound, upper_bound)
    
    current_time = time.time()
    simulated_angle = np.sin(current_time * 0.5) * 30 
    
    frame_processed_color = cv2.cvtColor(frame_processed_mono, cv2.COLOR_GRAY2BGR)
    return frame_processed_color, simulated_angle

# --- FUNGSI UTAMA WEBSOCKET SERVER (Diperbaiki) ---

async def serve_bs_connection(websocket):
    """Fungsi handler yang dieksekusi saat Base Station terhubung."""
    print(f"\n[CONNECTION] Base Station berhasil terhubung dari {websocket.remote_address}")
    
    # Cek apakah kamera gagal diinisialisasi di main()
    if cap is None or not cap.isOpened():
        error_msg = "FATAL: Kamera tidak terbuka. Hubungan ke BS ditutup."
        print(f"[ERROR] {error_msg}")
        await websocket.send(json.dumps({"type": "error", "message": error_msg}))
        return

    # Loop Pengiriman Data
    try:
        while True:
            start_time = time.time()
            
            # Gunakan lock karena cv2.read() bukan thread-safe
            async with cap_lock:
                ret, frame_raw = cap.read()
            
            # --- DEBUGGING LAMA DIHAPUS DARI SINI ---
            
            if not ret:
                await asyncio.sleep(0.01)
                continue
                
            height, width = frame_raw.shape[:2]
            
            # 2. Proses Citra & Hitung Sudut
            frame_processed, steering_angle = process_frame_and_get_angle(frame_raw)
            
            # 3. Encoding dan Pengiriman Asinkron
            msg_raw = encode_image(frame_raw, "raw", width, height)
            await websocket.send(msg_raw)
            
            msg_proc = encode_image(frame_processed, "processed", width, height)
            await websocket.send(msg_proc)

            msg_telemetry = create_telemetry_message(steering_angle, False)
            await websocket.send(msg_telemetry)
            
            # 4. Pengaturan FPS (Throttle)
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
    
    # 1. INIT KAMERA (DI SINI MASALAHNYA HARUS DIATASI)
    print("INFO: Mencoba menghubungkan ke IP Camera...")
    
    # *** PENTING: Menggunakan FFMPEG backend untuk stabilitas IP stream ***
    cap = cv2.VideoCapture(IP_CAM_URL, cv2.CAP_FFMPEG) 
    
    # Cek apakah kamera gagal dibuka
    if not cap.isOpened():
        print(f"FATAL ERROR: Gagal terhubung ke IP Camera di {IP_CAM_URL}.")
        print("Aksi: Cek IP/Port di browser. Cek Firewall. Program dihentikan.")
        return # Program berhenti jika kamera gagal dibuka

    print("SUCCESS: IP Camera berhasil dibuka.")
    
    # 2. MULAI SERVER WEBSOCKET
    print(f"INFO: Mencoba memulai server di ws://{SERVER_HOST}:{SERVER_PORT}")
    
    # --- KODE YANG SUDAH DIKOREKSI ---
    async with websockets.serve(
    serve_bs_connection, # <--- HANDLER SEBAGAI ARGUMEN POSISIONAL PERTAMA
    host=SERVER_HOST, 
    port=SERVER_PORT
    ):
        print("INFO: Server WebSocket berhasil dimulai.")
        print("Akses Base Station di browser, lalu koneksikan ke alamat di atas.")
        await asyncio.Future() # Tahan server agar tetap berjalan selamanya

if __name__ == "__main__":
    try:
        # Jalankan main loop
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n[STOP] Server dihentikan oleh pengguna (Ctrl+C).")
        
    except OSError as e:
        if "address already in use" in str(e):
            print(f"ERROR: Port {SERVER_PORT} sudah digunakan. Tutup program lain atau gunakan port yang berbeda.")
        else:
            raise
            
    finally:
        # PENTING: Lakukan cleanup kamera di akhir program
        if cap is not None and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows() 
        print("[CLEANUP] Kamera dan sumber daya ditutup.")
