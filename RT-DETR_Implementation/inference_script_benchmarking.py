import cv2
import time
import psutil
import subprocess
from ultralytics import RTDETR

# === Settings ===
VIDEO_PATH = "" #Input Video Path
MODEL_PATH = "rtdetr-l.pt"
OUTPUT_PATH = "" #Path to Save Output Video
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

# === Load Model ===
model = RTDETR(MODEL_PATH)

# === GPU Usage Helper ===
def get_gpu_usage():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return int(result.stdout.strip().split('\n')[0])
    except:
        return -1

# === Open Video ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_source = cap.get(cv2.CAP_PROP_FPS)

# === Setup Video Writer to save output ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_source, (WINDOW_WIDTH - 100, WINDOW_HEIGHT - 150))

# === Init counters ===
frame_count = 0
start_time = time.time()
second_start_time = time.time()
frames_this_second = 0

print("\n--- Real-Time Inference Benchmark ---\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start timing
    frame_start_time = time.time()

    # === Inference ===
    results = model(frame, verbose=False)
    annotated = results[0].plot()
    annotated = cv2.resize(annotated, (WINDOW_WIDTH - 100, WINDOW_HEIGHT - 150))

    # === Metrics ===
    frame_count += 1
    frames_this_second += 1
    current_time = time.time()

    cpu = psutil.cpu_percent(interval=0.01)
    gpu = get_gpu_usage()
    ram = psutil.virtual_memory().percent
    fps_live = frames_this_second

    # === Overlay on frame ===
    cv2.putText(annotated, f"Frame: {frame_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(annotated, f"FPS: {fps_live}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(annotated, f"CPU: {cpu:.1f}%", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    if gpu != -1:
        cv2.putText(annotated, f"GPU: {gpu:.1f}%", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), 2)
    cv2.putText(annotated, f"RAM: {ram:.1f}%", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 255), 2)

    # === Save and show ===
    out.write(annotated)
    cv2.imshow("RT-DETR Inference + Benchmark", annotated)

    # === Terminal log ===
    print(f"[Frame {frame_count}] FPS: {fps_live}, CPU: {cpu:.1f}%, GPU: {gpu:.1f}%, RAM: {ram:.1f}%")

    # Reset FPS counter every second
    if current_time - second_start_time >= 1:
        frames_this_second = 0
        second_start_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()

# === Final Summary ===
total_time = time.time() - start_time
avg_fps = frame_count / total_time

print("\n--- Inference Summary ---")
print(f"Total Frames Processed: {frame_count}")
print(f"Total Time Elapsed: {total_time:.2f} seconds")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Output Video Saved to: {OUTPUT_PATH}")
