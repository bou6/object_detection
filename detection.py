import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from time import sleep

# --- CONFIG ---
MODEL_PATH = "coco_model/detect.tflite"  # path to your tflite file

# Load labels from a file
LABELS = []
with open("coco_model/labelmap.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and line != "???":
            LABELS.append(line)
print(f"Loaded {len(LABELS)} labels")

# --- LOAD MODEL ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- MODEL INSPECTION ---
print("=== MODEL INPUT DETAILS ===")
for i, details in enumerate(input_details):
    print(f"Input {i}:")
    print(f"  Name: {details['name']}")
    print(f"  Shape: {details['shape']}")
    print(f"  Type: {details['dtype']}")
    print(f"  Index: {details['index']}")

print("\n=== MODEL OUTPUT DETAILS ===")
for i, details in enumerate(output_details):
    print(f"Output {i}:")
    print(f"  Name: {details['name']}")
    print(f"  Shape: {details['shape']}")
    print(f"  Type: {details['dtype']}")
    print(f"  Index: {details['index']}")

input_shape = input_details[0]['shape']  # e.g. [1, 224, 224, 3]
height, width = input_shape[1], input_shape[2]

print(f"\n=== PROCESSED INFO ===")
print(f"Input image size: {width}x{height}")
print(f"Number of output tensors: {len(output_details)}")


# --- INIT CAMERA ---
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Starting detection loop. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize and preprocess
        frame_resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]          # [N,4]
        class_ids = interpreter.get_tensor(output_details[1]['index'])[0]      # [N]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]         # [N]
        num = int(interpreter.get_tensor(output_details[3]['index'])[0])       # scalar

        # Print detections
        threshold = 0.5  # only show objects with confidence > 50%
        for i in range(num):
            class_id = int(class_ids[i])
            score = float(scores[i])
            if score > threshold:
                label = LABELS[class_id] if class_id < len(LABELS) else f"class {class_id}"
                # get the bounding box coordinates
                box = boxes[i]  # [ymin, xmin, ymax, xmax] normalized
                ymin, xmin, ymax, xmax = box
                ymin = int(max(1, ymin * frame.shape[0]))
                xmin = int(max(1, xmin * frame.shape[1]))
                ymax = int(min(frame.shape[0]-1, ymax * frame.shape[0]))
                xmax = int(min(frame.shape[1]-1, xmax * frame.shape[1]))
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                print(f"Detected: {label} ({score*100:.1f}%)")

        # Optional: display camera feed
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(0.1)  # small delay

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped detection loop.")
