import cv2
from ultralytics import YOLO

# Load the YOLOv8 segmentation model (e.g., yolov8n-seg.pt, yolov8s-seg.pt)
model = YOLO("yolov8s-seg.pt")  # or use a path to your custom weights

# Set the class name for person (COCO ID 0)
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.4

# Load image
image_path = "Roadzone.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Run inference
results = model(image)[0]  # YOLOv8 returns a list; we take the first item

# Iterate through detections
for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
    if int(cls) == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
        x1, y1, x2, y2 = map(int, box)
        label = f"Person {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show result
# Resize image for display
display_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)  # 50% smaller
cv2.imshow("Person Detection", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
