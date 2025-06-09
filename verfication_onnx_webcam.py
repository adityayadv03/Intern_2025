import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("yolov7.onnx")
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

# Start webcam
cap = cv2.VideoCapture(0)

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_orig, w_orig = frame.shape[:2]
    image_resized = cv2.resize(frame, (640, 640))
    input_image = image_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

    # Inference
    outputs = session.run(None, {input_name: input_image})

    # Combine outputs
    predictions = np.concatenate([o.reshape(-1, 85) for o in outputs], axis=0)
    predictions = predictions[predictions[:, 4] > CONF_THRESHOLD]

    boxes, scores, class_ids, boxes_for_nms = [], [], [], []

    if len(predictions) > 0:
        boxes_raw = predictions[:, :4]
        confs = predictions[:, 4]
        class_probs = predictions[:, 5:]
        class_ids_raw = class_probs.argmax(axis=1)
        scores_raw = confs * class_probs.max(axis=1)

        # Convert boxes cxcywh to xyxy
        cx = boxes_raw[:, 0]
        cy = boxes_raw[:, 1]
        w = boxes_raw[:, 2]
        h = boxes_raw[:, 3]
        x1 = (cx - w / 2) * w_orig / 640
        y1 = (cy - h / 2) * h_orig / 640
        x2 = (cx + w / 2) * w_orig / 640
        y2 = (cy + h / 2) * h_orig / 640

        for i in range(len(predictions)):
            box = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
            boxes.append(box)
            boxes_for_nms.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
            scores.append(float(scores_raw[i]))
            class_ids.append(int(class_ids_raw[i]))

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, CONF_THRESHOLD, NMS_THRESHOLD)

        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            label = f"{coco_classes[class_ids[i]]}: {scores[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show result
    cv2.imshow("YOLOv7 ONNX - Webcam", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

