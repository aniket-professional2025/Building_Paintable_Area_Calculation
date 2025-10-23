# Importing required libraries
import cv2
import numpy as np
import os

# Calculate_scale_factor function
def calculate_scale_factor(ref_pixels, ref_actual_feet):
    return ref_actual_feet / ref_pixels

# Modified measure_building_dimensions with Manual ROI
def measure_building_dimensions_with_roi(image_path, scale_factor, roi_coords = None, output_path = None):

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at path: {image_path}")
        return None, None

    # Apply ROI if provided
    if roi_coords:
        x_start, y_start, x_end, y_end = roi_coords
        # Ensure ROI is within image bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(img.shape[1], x_end)
        y_end = min(img.shape[0], y_end)

        img_roi = img[y_start:y_end, x_start:x_end]
        if img_roi.shape[0] == 0 or img_roi.shape[1] == 0:
            print("Error: ROI resulted in an empty image region. Adjust ROI coordinates.")
            return None, None
        
        # Process the ROI image
        gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        edges_roi = cv2.Canny(gray_roi, 50, 150, apertureSize=3)
        
        y_coords_roi, x_coords_roi = np.where(edges_roi > 0)

        if len(x_coords_roi) == 0:
            print("Error: No edges detected within ROI. Adjust Canny thresholds or ROI.")
            return None, None
            
        # Get min/max within the ROI, then offset back to original image coordinates
        x_min_roi, x_max_roi = np.min(x_coords_roi), np.max(x_coords_roi)
        y_min_roi, y_max_roi = np.min(y_coords_roi), np.max(y_coords_roi)
        
        # Final bounding box coordinates in the original image space
        x_min_final = x_min_roi + x_start
        x_max_final = x_max_roi + x_start
        y_min_final = y_min_roi + y_start
        y_max_final = y_max_roi + y_start

        building_width_pixels = x_max_final - x_min_final
        building_height_pixels = y_max_final - y_min_final

    else: # Original logic if no ROI is provided
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        y_coords, x_coords = np.where(edges > 0)
        
        if len(x_coords) == 0:
            print("Error: No edges detected. Adjust Canny thresholds.")
            return None, None
            
        x_min_final, x_max_final = np.min(x_coords), np.max(x_coords)
        y_min_final, y_max_final = np.min(y_coords), np.max(y_coords)
        
        building_width_pixels = x_max_final - x_min_final
        building_height_pixels = y_max_final - y_min_final

    # Visualization (Drawing the final bounding box)
    cv2.rectangle(img, (x_min_final, y_min_final), (x_max_final, y_max_final), (0, 0, 255), 2)

    # SAVE THE IMAGE
    if output_path:
        try:
            cv2.imwrite(output_path, img)
            print(f"Successfully saved image with bounding box to: {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
    
    window_name = "Measured Section (Bounding Box with ROI)" if roi_coords else "Measured Section (Global Bounding Box)"
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    height_feet = building_height_pixels * scale_factor
    width_feet = building_width_pixels * scale_factor
    
    return height_feet, width_feet

# --- EXAMPLE USAGE ---

REF_ACTUAL_HEIGHT_FT = 45.0  # Actual height of the reference object in feet
REF_PIXELS = 408.397 # You'd need to measure this for the reference object *within your ROI*

scale = calculate_scale_factor(REF_PIXELS, REF_ACTUAL_HEIGHT_FT)
print(f"Calculated Scale Factor: {scale:.4f} feet/pixel") 

IMAGE_PATH = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_8.jpg"
# OUTPUT_PATH = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Output_with_roi.image_8.jpg"

# x_start: around 100-150 pixels from left
# y_start: around 50-100 pixels from top (below the sky boundary)
# x_end: around 850-900 pixels from left (before the far right edge of the frame)
# y_end: around 900-950 pixels from top (above the deep ground clutter)

# Let's assume these values based on visual inspection of your image:
REFINED_ROI = (60,200,980,700) # (60,200,980,850)

print("\n--- Measuring with Manual ROI ---")
height_refined, width_refined = measure_building_dimensions_with_roi(IMAGE_PATH, scale, roi_coords = REFINED_ROI)

if height_refined is not None and width_refined is not None:
    print(f"Estimated Building Height (Refined ROI): {height_refined:.2f} ft")
    print(f"Estimated Building Width (Refined ROI): {width_refined:.2f} ft")