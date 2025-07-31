import cv2

# Load original image
original_img = cv2.imread("Roadzone.jpg")

# Load modified cropped images
cropped_images = {
    1: cv2.imread("cropped_persons/person_1.jpg"),
    2: cv2.imread("cropped_persons/person_2.jpg"),
    3: cv2.imread("cropped_persons/person_3.jpg")
}

# Bounding box coordinates
bounding_boxes = {
    1: (614, 418, 747, 705),
    2: (787, 116, 895, 369),
    3: (502, 2, 629, 248)
}

# For each version, skip one person and replace the other two
for skip_id in [1, 2, 3]:
    image_copy = original_img.copy()
    for pid, (x1, y1, x2, y2) in bounding_boxes.items():
        if pid == skip_id:
            continue  # Skip replacing this one

        # Resize cropped image to fit bounding box (in case edited dimensions changed)
        modified = cv2.resize(cropped_images[pid], (x2 - x1, y2 - y1))
        image_copy[y1:y2, x1:x2] = modified

    # Save the result
    output_filename = f"partial_replacement_{skip_id}.jpg"
    cv2.imwrite(output_filename, image_copy)
    print(f"Saved: {output_filename} (Person {skip_id} was NOT replaced)")
