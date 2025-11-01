import asyncio
import websockets
import json
import socket
import threading
import time

SERVER_HOST = '0.0.0.0' # Station Server akan mendengarkan di semua interface
WEBSOCKET_PORT = 8081 # Port untuk Base Station (WebSocket)
UDP_PORT = 8080   # Port untuk Robot Program (UDP Listener)
FRAME_RATE_LIMIT = 15 # Batasan FPS untuk pengiriman ke Base Station

# variabel globa yg akan diisi udp listener
GLOBAL_ROBOT_DATA = {
    "timestamp": 0.0,
    "steering_angle": 0.0, 
    "laneStatus": "Lost", 
    "speed": 0.0,
    "deviation": 0.0, 
    "obstacleDetected": False,
    "obstacleDistance": 0.0,
    "obstaclePosition": "unknown",
    "raw_image_b64": "",     # Base64 Image Raw dari Robot
    "processed_image_b64": "" # Base64 Image Processed dari Robot
}

# UDP listener

def udp_listener_task():
    """Terus mendengarkan paket UDP dari Robot Program dan mengupdate GLOBAL_ROBOT_DATA."""
    global GLOBAL_ROBOT_DATA
   
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((SERVER_HOST, UDP_PORT))
        print(f"[UDP] Siap menerima data dari Robot di {SERVER_HOST}:{UDP_PORT}")
    except Exception as e:
        print(f"[FATAL UDP ERROR] Gagal bind socket: {e}")
        return

    while True:
        try:
            # Tunggu paket data
            data, addr = sock.recvfrom(65507) # Max UDP packet size
   
            # Data Diharapkan dalam format JSON string
            data_json = json.loads(data.decode('utf-8'))
 
            # --- UPDATE GLOBAL STATE ---
            if 'data' in data_json:
                # Ini mengasumsikan Robot mengirimkan satu paket besar berisi semua data
                GLOBAL_ROBOT_DATA.update(data_json['data'])
                GLOBAL_ROBOT_DATA['timestamp'] = time.time()

        except json.JSONDecodeError:
            print("[UDP WARNING] Menerima data non-JSON atau korup.")
        except Exception as e:
            print(f"[UDP ERROR] Kesalahan tak terduga: {e}")
            time.sleep(0.001)

# --- WEBSOCKET SERVER (Main Task) ---

async def serve_bs_connection(websocket):
    """Handler WebSocket untuk Base Station."""
    print(f"\n[CONNECTION] Base Station terhubung dari {websocket.remote_address}")

    try:
        last_send_time = time.time()

        while True:
            # 1. Hitung jeda (throttling)
            elapsed_time = time.time() - last_send_time
            delay = (1.0 / FRAME_RATE_LIMIT) - elapsed_time
            if delay > 0:
                await asyncio.sleep(delay)
            last_send_time = time.time()
 
            # 2. Kumpulkan data terbaru dari GLOBAL_ROBOT_DATA
            data_to_send = GLOBAL_ROBOT_DATA.copy()

            # 3. Kirim Frame Raw (jika ada)
            if data_to_send["raw_image_b64"]:
                msg_raw = json.dumps({
                    "type": "image_raw", 
                    "data": data_to_send.pop("raw_image_b64"),
                    "width": 640, "height": 480 # Asumsi
                })
                await websocket.send(msg_raw)

            # 4. Kirim Frame Processed (jika ada)
            if data_to_send["processed_image_b64"]:
                msg_proc = json.dumps({
                    "type": "image_processed", 
                    "data": data_to_send.pop("processed_image_b64"),
                    "width": 640, "height": 480 # Asumsi
                })
                await websocket.send(msg_proc)

            # 5. Kirim Telemetri
            # Membuang kunci gambar dari data telemetri agar bersih
            data_telemetry = {k: v for k, v in data_to_send.items() if not k.endswith("_b64") and k != 'timestamp'}
            msg_telemetry = json.dumps({
                "type": "telemetry",
                "data": data_telemetry
            })
            await websocket.send(msg_telemetry)

    except websockets.exceptions.ConnectionClosed:
        print(f"[DISCONNECT] Base Station terputus: {websocket.remote_address}")
    except Exception as e:
        print(f"[ERROR] Kesalahan saat beroperasi: {e}")

async def main():
    """Fungsi inisialisasi dan memulai Server."""
    print("--- STATION SERVER (UDP Listener & WebSocket Server) ---")

    # 1. MULAI THREAD UDP LISTENER
    udp_thread = threading.Thread(target=udp_listener_task, daemon=True)
    udp_thread.start()

    # 2. MULAI SERVER WEBSOCKET
    print(f"INFO: Mencoba memulai WebSocket Server di ws://{SERVER_HOST}:{WEBSOCKET_PORT}")

    async with websockets.serve(serve_bs_connection, host=SERVER_HOST, port=WEBSOCKET_PORT):
        print("INFO: Server WebSocket berhasil dimulai.")
        await asyncio.Future() # Tahan server agar tetap berjalan

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] Server dihentikan oleh pengguna (Ctrl+C).")
    except OSError as e:
        if "address already in use" in str(e):
            print(f"ERROR: Port sudah digunakan. Pastikan tidak ada program lain yang berjalan di {WEBSOCKET_PORT}.")
        else:
            raise
