import cv2
import os
from ultralytics import YOLO

# Load model
model = YOLO("yolov8s-seg.pt")
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.4

# Load image
image_path = "Roadzone.jpg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Run detection
results = model(image)[0]

# Create output folder
os.makedirs("cropped_persons", exist_ok=True)

# Process each detection
for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
    if int(cls) == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
        x1, y1, x2, y2 = map(int, box)

        # Calculate padding (20%)
        box_width = x2 - x1
        box_height = y2 - y1
        pad_w = int(box_width * 0.2)
        pad_h = int(box_height * 0.2)

        # Expand box with clipping to image borders
        x1_pad = max(x1 - pad_w, 0)
        y1_pad = max(y1 - pad_h, 0)
        x2_pad = min(x2 + pad_w, width - 1)
        y2_pad = min(y2 + pad_h, height - 1)

        # Crop and save
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        crop_filename = f"cropped_persons/person_{i+1}.jpg"
        cv2.imwrite(crop_filename, cropped)

        # Print pixel coordinates for future replacement
        print(f"Person {i+1}:")
        print(f"  Padded Box: x1={x1_pad}, y1={y1_pad}, x2={x2_pad}, y2={y2_pad}")
        print(f"  Saved to: {crop_filename}")
        print("-" * 40)
