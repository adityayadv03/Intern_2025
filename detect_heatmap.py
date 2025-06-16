import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from sort import Sort
import os
import sys
from pathlib import Path

# === CONFIG ===
ENGINE_PATH = "yolov7_fp16.engine"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "inference/Video/People.mp4"
OUTPUT_DIR = "runs/detect/Heat Map/Output1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_W, INPUT_H = 640, 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# COCO classes
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
           "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
           "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, :]
    return img

def infer(engine, context, stream, bindings, inputs, outputs, frame):
    input_data = preprocess(frame)
    np.copyto(inputs[0][0], input_data.ravel())

    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    return outputs[0][0].reshape(-1, 85)

def postprocess(predictions, frame_shape):
    predictions = predictions[predictions[:, 4] > CONF_THRESH]
    if predictions.shape[0] == 0:
        return []

    scores = predictions[:, 4] * predictions[:, 5:].max(axis=1)
    class_ids = predictions[:, 5:].argmax(axis=1)
    boxes = predictions[:, :4]

    h, w = frame_shape[:2]
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - bw / 2) * w / INPUT_W
    y1 = (cy - bh / 2) * h / INPUT_H
    x2 = (cx + bw / 2) * w / INPUT_W
    y2 = (cy + bh / 2) * h / INPUT_H
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)

    final_boxes = []
    for i in range(len(boxes)):
        if CLASSES[class_ids[i]] != 'person':
            continue
        final_boxes.append([*boxes[i], scores[i]])
    return final_boxes

def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding))
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    cap = cv2.VideoCapture(VIDEO_PATH)
    tracker = Sort()

    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read video.")
        return

    height, width = frame.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    last_frame = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("[INFO] Generating heatmap...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()

        detections = infer(engine, context, stream, bindings, inputs, outputs, frame)
        people = postprocess(detections, frame.shape)
        tracks = tracker.update(np.array(people))

        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if 0 <= cx < width and 0 <= cy < height:
                heatmap[cy, cx] += 3

    # Create blurred and normalized heatmap
    blurred = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(last_frame, 0.6, heatmap_color, 0.4, 0)

    output_path = str(Path(OUTPUT_DIR) / "people_heatmap_overlay.png")
    cv2.imwrite(output_path, overlay)
    print(f"[DONE] Saved overlay heatmap image: {output_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

