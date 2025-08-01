import cv2

# Load the image
image_path = "Roadzone.jpg"
image = cv2.imread(image_path)

# Define bounding boxes for 5 humans (x1, y1, x2, y2)
bounding_boxes = [
    (262, 200, 338, 330),  # Person 1 (Excavator driver)
    (550, 132, 612, 290),  # Person 2 (behind excavator arm)
    (698, 226, 762, 358),  # Person 3 (far right)
    (538, 380, 601, 511),  # Person 4 (bottom center)
    (99, 102, 144, 177),   # Person 5 (far left)
]

# Draw bounding boxes on the image
for (x1, y1, x2, y2) in bounding_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

# Save or display the output
output_path = "Roadzone_with_boxes.jpg"
cv2.imwrite(output_path, image)
print(f"Image saved to: {output_path}")

# Optional: display the image (uncomment below lines if running locally)
# cv2.imshow("Bounding Boxes", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
