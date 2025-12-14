import cv2
import numpy as np

def apply_filter(image, filter_type):
    # Create a copy of the image to avoid modifying the original
    filtered_image = image.copy()

    if filter_type == "red_tint":
        filtered_image[:, :, 0] = 0   # Remove Blue channel
        filtered_image[:, :, 1] = 0   # Remove Green channel

    elif filter_type == "green_tint":
        filtered_image[:, :, 0] = 0   # Remove Blue channel
        filtered_image[:, :, 2] = 0   # Remove Red channel

    elif filter_type == "blue_tint":
        filtered_image[:, :, 1] = 0   # Remove Green channel
        filtered_image[:, :, 2] = 0   # Remove Red channel

    elif filter_type == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif filter_type == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered_image = cv2.transform(image, kernel)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    elif filter_type == "canny":
        edges = cv2.Canny(image, 100, 200)
        filtered_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return filtered_image


# Load the image
image_path = "sample1.jpg"   # 👉 Put your image name or full path here
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found")
else:
    filter_type = "original"

    print("Press the following keys to apply filters:")
    print(" r : Red Tint")
    print(" g : Green Tint")
    print(" b : Blue Tint")
    print(" s : Sepia")
    print(" c : Canny Edge Detection")
    print(" y : Gray")
    print(" q : Quit")

    while True:
        filtered_image = apply_filter(image, filter_type)
        cv2.imshow("Filtered Image", filtered_image)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('r'):
            filter_type = "red_tint"
        elif key == ord('g'):
            filter_type = "green_tint"
        elif key == ord('b'):
            filter_type = "blue_tint"
        elif key == ord('s'):
            filter_type = "sepia"
        elif key == ord('c'):
            filter_type = "canny"
        elif key == ord('y'):
            filter_type = "gray"
        elif key == ord('q'):
            break
        else:
            print("Invalid key! Please use r, g, b, s, c, y or q")

    cv2.destroyAllWindows()
