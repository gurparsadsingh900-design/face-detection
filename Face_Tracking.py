import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
# This is a standard classifier for frontal faces.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture (use webcam)
# '0' typically refers to the default primary camera.
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale (improves face detection performance)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    # The detectMultiScale function returns a list of (x, y, w, h) bounding boxes.
    # scaleFactor=1.1: Reduces image size by 10% at each step.
    # minNeighbors=5: A higher value requires more neighborhood hits for a detection.
    # minSize=(30, 30): Minimum size of the face to be detected.
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        # BGR color (255, 0, 0) is Blue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the count of faces
    face_count = len(faces)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'People Count: {face_count}'

    # Put the text on the frame
    # cv2.putText(image, text, position, font, font_scale, color, thickness, line_type)
    cv2.putText(frame, text, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with face detection and people count
    cv2.imshow('Face Tracking and Counting', frame)

    # Exit the loop when the 'q' key is pressed
    # cv2.waitKey(1) waits for 1 millisecond.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()