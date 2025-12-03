import cv2

def capture_image(output_path="capture.jpg", camera_index=1):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise Exception("Could not open video device. Check camera connection and index.")

    # Read one frame
    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise Exception("Failed to capture image from camera.")

    # Save as JPEG
    cv2.imwrite(output_path, frame)

    # Release camera
    cap.release()

    print(f"Image saved to {output_path}")

# Run it
capture_image()
