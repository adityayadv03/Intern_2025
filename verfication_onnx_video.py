import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model on CPU
session = ort.InferenceSession("yolov7.onnx", providers=["CPUExecutionProvider"])
#Load ONNX model on GPU
#session = ort.InferenceSession("yolov7.onnx", providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name

# COCO class labels
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Open video file
cap = cv2.VideoCapture("inference/Video/Test.mp4")

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

conf_threshold = 0.25
nms_threshold = 0.45

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_orig, w_orig = frame.shape[:2]

    # Resize and normalize
    image_resized = cv2.resize(frame, (640, 640))
    input_image = image_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

    # Run inference
    outputs = session.run(None, {input_name: input_image})
    predictions = np.concatenate([o.reshape(-1, 85) for o in outputs], axis=0)

    predictions = predictions[predictions[:, 4] > conf_threshold]

    if len(predictions) > 0:
        boxes = predictions[:, :4]
        scores = predictions[:, 4] * predictions[:, 5:].max(axis=1)
        class_ids = predictions[:, 5:].argmax(axis=1)

        # Convert boxes from center xywh to corner xyxy
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes[:, 0] = cx - w / 2
        boxes[:, 1] = cy - h / 2
        boxes[:, 2] = cx + w / 2
        boxes[:, 3] = cy + h / 2

        # Scale to original image
        scale_w = w_orig / 640
        scale_h = h_orig / 640
        boxes[:, [0, 2]] *= scale_w
        boxes[:, [1, 3]] *= scale_h
        boxes = boxes.astype(int)

        boxes_for_nms = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), conf_threshold, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                label = f"{coco_classes[class_ids[i]]}: {scores[i]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("YOLOv7 - Video Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Output video saved as 'output_video.mp4'")
