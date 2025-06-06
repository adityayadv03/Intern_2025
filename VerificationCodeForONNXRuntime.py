import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("yolov7.onnx")
input_name = session.get_inputs()[0].name

# Load and preprocess image
original_image = cv2.imread("inference/images/image3.jpg")
h_orig, w_orig = original_image.shape[:2]

# Resize + normalize
image_resized = cv2.resize(original_image, (640, 640))
input_image = image_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

# Run inference
outputs = session.run(None, {input_name: input_image})

# Concatenate outputs
predictions = np.concatenate([o.reshape(-1, 85) for o in outputs], axis=0)

# Confidence threshold
conf_threshold = 0.1
predictions = predictions[predictions[:, 4] > conf_threshold]

if len(predictions) == 0:
    print("⚠️ No detections above confidence threshold.")
else:
    # Extract boxes, scores, and class ids
    boxes = predictions[:, :4]
    scores = predictions[:, 4] * predictions[:, 5:].max(axis=1)
    class_ids = predictions[:, 5:].argmax(axis=1)

    # Convert boxes from center xywh to corner xyxy correctly
    cx = boxes[:, 0].copy()
    cy = boxes[:, 1].copy()
    w = boxes[:, 2].copy()
    h = boxes[:, 3].copy()

    boxes[:, 0] = cx - w / 2  # x1
    boxes[:, 1] = cy - h / 2  # y1
    boxes[:, 2] = cx + w / 2  # x2
    boxes[:, 3] = cy + h / 2  # y2

    # Scale boxes to original image size
    scale_w = w_orig / 640
    scale_h = h_orig / 640
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    boxes = boxes.astype(int)

    # Prepare boxes in x,y,w,h format for NMS
    boxes_for_nms = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_for_nms.append([x1, y1, x2 - x1, y2 - y1])

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), score_threshold=conf_threshold, nms_threshold=0.45)

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

    # Draw boxes on the original image
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            label = f"{coco_classes[class_ids[i]]}: {scores[i]:.2f}"
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or show the output
    cv2.imwrite("output.jpg", original_image)
    print("✅ Detections saved to 'output.jpg'")
