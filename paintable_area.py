# Importing Required Packages
from objects_area import detect_objects
from total_building_area import building_information 
import numpy as np
import cv2
import torch
from PIL import Image

# Define the main function to calculate the paintable area
def paintable_area(image_path: str, floor_tolerence: int = 130, actual_floor_height_feet: float = 10.0, real_object_height_feet: float = 3.5,  detection_threshold: float = 0.5, output_path: str = None):

    """
    Calculates the total building area, object (window) area, and the final 
    paintable area (Building Area - Object Area) for a given image.

    Args:
        image_path (str): Path to the input image.
        floor_tolerence (int): Tolerance for clustering horizontal lines into floors.
        actual_floor_height (float): Actual height of one floor in feet.
        real_object_height_feet (float): Known real-world height of a reference object (e.g., a window) in feet, used for scaling.
        detection_threshold (float): Confidence threshold for object detection.
        output_path (str): Path to save the final annotated image.
        
    Returns:
        tuple: (num_floors, real_height, real_width, total_building_area, total_object_area, paintable_area)
    """

    # --- 1. Calculate Building Information ---
    print("\n--- 1. Calculating Building Information ---")
    
    num_floors, real_height, real_width, total_building_area = building_information(image_path = image_path, tolerence = floor_tolerence, actual_floor_height = actual_floor_height_feet, output_path = None)

    print(f"Building Information: Floors = {num_floors}, Height = {real_height:.0f} ft, Width = {real_width:.0f} ft, Area = {total_building_area:.0f} sq. ft")
    
    # --- 2. Calculate Object (Window) Area ---
    print("\n--- 2. Calculating Object (Window) Area ---")
    
    total_object_area = detect_objects(image_path = image_path, real_height_feet = real_object_height_feet, threshold = detection_threshold, output_path = None)
    total_object_area = float(total_object_area)
    print(f"Total Object Area (Windows): {total_object_area:.0f} sq. feet")

    # --- 3. Calculate Paintable Area ---
    paintable_area_val = total_building_area - total_object_area
    paintable_area_val = np.round(paintable_area_val, 0)
    print(f"\n[FINAL RESULT] Paintable Area: {paintable_area_val:.0f} sq. feet")

    # --- 4. Annotate and Save Final Image ---
    print("\n--- 4. Annotating and Saving Final Image ---")
    try:
        img_np = cv2.imread(image_path)
        if img_np is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
    except Exception:
        raise FileNotFoundError("Could not load image for final annotation.")
    
    # Define Annotation Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 0, 0) # Blue color for text
    thickness = 2
    
    # Text to be written on the image
    text_lines = [
        f"Estimated Floors: {num_floors}",
        f"Estimated Height: {np.round(real_height, 0):.0f} Feets",
        f"Estimated Width: {np.round(real_width, 0):.0f} Feets",
        f"Total Building Area: {np.round(total_building_area, 0):.0f} Sq. Feet",
        f"Total Window Area: {np.round(total_object_area, 0):.0f} Sq. Feet",
        f"Paintable Area: {np.round(paintable_area_val, 0):.0f} Sq. Feet"
    ]
    
    # Put text on image (defining the Coordinates)
    y_start = 40
    line_spacing = 30
    for i, line in enumerate(text_lines):
        cv2.putText(img_np, line, (10, y_start + i * line_spacing), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the final image
    if output_path:
        cv2.imwrite(output_path, img_np)
        print(f"Final annotated image saved to: {output_path}")
        
    # Display the image (optional/comment out if running in headless environment)
    cv2.imshow("Paintable Area Result", img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 5. Return Results ---
    return (
        num_floors, 
        np.round(real_height, 0), 
        np.round(real_width, 0), 
        total_building_area, 
        total_object_area, 
        paintable_area_val
    )

# Inference Example
if __name__ == "__main__":

    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_8.jpg"
    output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_8_Final_Result.jpg"

    print("\nStarting Paintable Area Calculation...")
    
    results = paintable_area(
        image_path = image_path, 
        floor_tolerence = 130, 
        actual_floor_height_feet = 10.0,
        real_object_height_feet = 3.5, 
        detection_threshold = 0.5,
        output_path = output_path
    )
    
    num_floors, height, width, building_area, object_area, final_paintable_area = results
    
    print("\n--- Function Return Values ---")
    print(f"Number of Floors: {num_floors}")
    print(f"Building Height: {height:.2f} Feets")
    print(f"Building Width: {width:.2f} Feets")
    print(f"Total Building Area: {building_area:.2f} Sq. Feet")
    print(f"Total Object Area (Windows): {object_area:.2f} Sq. Feet")
    print(f"Final Paintable Area: {final_paintable_area:.2f} Sq. Feet")