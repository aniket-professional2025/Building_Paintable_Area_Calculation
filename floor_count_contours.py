# Importing required packages
import cv2
import numpy as np

# LFunction to count floors based on contours
def count_floors(image_path,
    blur_ksize = (5, 5),
    block_size = 21,
    C = 10,
    morph_kernel_size = (5, 5),
    morph_iterations = 2,
    aspect_ratio_range = (0.5, 2.0),
    area_range = (1000, 20000),
    floor_tolerance = 40,
    resize_width = 800,
    show_result = True):

    """
    Count floors in a building image using classical computer vision (no deep learning).
    Parameters:
    -----------
    image_path : str --> Path to input image
    blur_ksize : tuple(int, int) --> Gaussian blur kernel size
    block_size : int --> Block size for adaptive threshold (must be odd)
    C : int --> Constant subtracted in adaptive thresholding
    morph_kernel_size : tuple(int, int) --> Kernel size for morphological closing
    morph_iterations : int --> Number of morphological iterations
    aspect_ratio_range : tuple(float, float) --> Acceptable (min, max) window width/height ratio
    area_range : tuple(int, int) --> Acceptable (min, max) contour area for window candidates
    floor_tolerance : int --> Pixel distance to group windows into the same floor
    resize_width : int --> Width to resize image for consistency
    show_result : bool --> Whether to display the output image
    """

    # Load the Image
    try:
        img = cv2.imread(image_path)
    except Exception as e:
        raise FileNotFoundError("Image Not Found")
    
    # Resize the Image for Consistency
    height, width = img.shape[:2]
    scale = resize_width / width
    img = cv2.resize(img, (int(width * scale), int(height * scale)))

    # Convert the Image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE on the gray image
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
    gray = clahe.apply(gray)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, blur_ksize , 0)

    # Apply Adaptive Thresholding to Isolate windows ADAPTIVE_THRESH_MEAN_C
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)

    # Morphological operations: Remove small noise and connect window regions
    kernel = np.ones(morph_kernel_size, np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = morph_iterations)

    # Contour detection
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculating the windows
    window_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        # Filter based on typical window shape and size
        if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and area_range[0] < area < area_range[1]:
            window_boxes.append((x, y, w, h))

    # Cluster windows by Y-coordinate (floor-wise): Sort windows from top to bottom
    window_boxes = sorted(window_boxes, key = lambda b: b[1])

    y_coords = [b[1] + b[3] // 2 for b in window_boxes]  # center Y of each window

    floor_groups = []

    # pixel threshold between floors
    for y in y_coords:
        if not floor_groups or abs(y - np.mean(floor_groups[-1])) > floor_tolerance:
            floor_groups.append([y])
        else:
            floor_groups[-1].append(y)

    num_floors = len(floor_groups)

    # # Draw results
    output = img.copy()
    # for (x, y, w, h) in window_boxes:
    #     cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Label each detected floor level
    for i, group in enumerate(floor_groups, 1):
        avg_y = int(np.mean(group))
        # cv2.putText(output, f'Floor {i}', (10, avg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(output, f"Estimated Floors: {num_floors}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    print("Estimated number of floors:", num_floors)

    # Display
    if show_result:
        cv2.imshow("Detected Windows and Floors", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return the floor number and the output
    return num_floors, output

# Inference on the function
# r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_8.jpg"
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_8.jpg"
    floors, output = count_floors(
        image_path,
        blur_ksize = (3,3),
        block_size = 15,
        C = 5,
        morph_kernel_size = (7, 7),
        morph_iterations = 3,
        aspect_ratio_range = (0.3, 3.0),
        area_range = (400, 40000),
        floor_tolerance = 30,
        resize_width = 900,
        show_result = True
    )