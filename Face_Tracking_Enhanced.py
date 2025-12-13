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

# Create a named window and set it to be resizable with a larger default size
window_name = 'Face Tracking and Counting'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
cv2.resizeWindow(window_name, 1920, 1080)  # Set initial size to Full HD (width, height)


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

    # Draw ellipses around the detected faces
    for (x, y, w, h) in faces:
        # Calculate center of the face
        center = (x + w // 2, y + h // 2)
        
        # Option 1: Draw an ellipse (oval) that surrounds the face
        # cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)
        # axes: (horizontal_radius, vertical_radius)
        axes = (w // 2, h // 2)  # Use half width and half height as radii
        cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 0, 0), 2)
        
        # Option 2 (Alternative): Draw a circle that surrounds the face
        # Uncomment the line below and comment out the ellipse above if you prefer a circle
        radius = max(w, h) // 2
        cv2.circle(frame, center, radius, (255, 0, 0), 2)

    # Display the count of faces with beautiful styling
    face_count = len(faces)
    
    # Get frame dimensions for responsive sizing
    frame_height, frame_width = frame.shape[:2]
    
    # Font settings - using DUPLEX for a cleaner, more elegant look
    # Make font scale responsive based on frame height
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = frame_height / 400  # Scales with window size (adjust divisor for size preference)
    font_thickness = max(1, int(frame_height / 250))  # Responsive thickness
    text_color = (255, 200, 0)  # Cyan-blue color (BGR format)
    bg_color = (50, 50, 50)  # Dark gray background
    padding = int(frame_height / 50)  # Responsive padding
    line_spacing = int(frame_height / 12)  # Responsive spacing between lines
    
    # Text content
    text1 = f'People Count: {face_count}'
    text2 = 'Press Q to Quit'
    
    # Get text sizes for background boxes
    (text1_width, text1_height), baseline1 = cv2.getTextSize(text1, font, font_scale, font_thickness)
    (text2_width, text2_height), baseline2 = cv2.getTextSize(text2, font, font_scale, font_thickness)
    
    # Position for first text (responsive)
    text1_x = int(frame_width / 60)
    text1_y = int(frame_height / 12)
    
    # Draw background rectangle for first text
    cv2.rectangle(frame, 
                  (text1_x - padding, text1_y - text1_height - padding),
                  (text1_x + text1_width + padding, text1_y + baseline1 + padding),
                  bg_color, -1)
    
    # Draw first text
    cv2.putText(frame, text1, (text1_x, text1_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Position for second text (with spacing)
    text2_x = int(frame_width / 60)
    text2_y = text1_y + line_spacing
    
    # Draw background rectangle for second text
    cv2.rectangle(frame,
                  (text2_x - padding, text2_y - text2_height - padding),
                  (text2_x + text2_width + padding, text2_y + baseline2 + padding),
                  bg_color, -1)
    
    # Draw second text
    cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Display the frame with face detection and people count
    cv2.imshow(window_name, frame)

    # Exit the loop when the 'q' key is pressed
    # cv2.waitKey(1) waits for 1 millisecond.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
