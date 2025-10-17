# Importing required packages
import numpy as np
import cv2

# The function to count floors based on Visible Lines
def count_floors_line(image_path, tolerence: int, output_path: str):

    # Load the image
    try:
        img = cv2.imread(image_path)
    except Exception as e:
        raise FileNotFoundError("Image not found")
    
    # Convert the image into gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Finding the Edges
    edges = cv2.Canny(blur, 50, 150)

    # Detect Lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 100, minLineLength = 100, maxLineGap = 10)

    # Creating the horizontal Lines
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10:
            horizontal_lines.append((y1 + y2) // 2)

    # Cluster Similar Lines
    horizontal_lines = sorted(horizontal_lines)
    floor_lines = []
    for y in horizontal_lines:
        if not floor_lines or abs(y - floor_lines[-1]) > tolerence:
            floor_lines.append(y)

    # Put the Answer on the image
    cv2.putText(img, f"Estimated Floors:{len(floor_lines)}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, img, params = None)

    cv2.imshow("Detected Floors", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return the result
    return floor_lines, len(floor_lines)

# The inference on the images
# Tolerence value 400 for Image_3.jpg
# Tolerence value 70 for Image_7.jpg
# Tolerence value 130 for Image_8.jpg
# Tolerence value 350 for Image_21.jpg
# Tolerence value 850 for Image_13.jpg

if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_13.jpg"
    output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_13_Result.jpg"
    floor_lines, floor_num = count_floors_line(image_path, tolerence = 850, output_path = output_path)
    print("The Floor Line is:", floor_lines)
    print("The number of Floors in the image is:", floor_num)