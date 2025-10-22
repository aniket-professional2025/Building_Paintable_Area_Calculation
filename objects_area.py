# Importing Required Packages
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2

# Setting the Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[DEBUG] The device is:", device)

# Prepare processor and model
model_id = "rziga/mm_grounding_dino_large_all"
print("[DEBUG] Setting the Model Name")

processor = AutoProcessor.from_pretrained(model_id)
print("[DEBUG] The processor is Loaded Successfully")

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("[DEBUG] The Model is Loaded Successfully")

# Function to detect windows using image_path and threshold
def detect_objects(image_path: str, real_height_feet: float = 3.5, threshold: float = 0.4, output_path = None):

    # Reading the Image
    try:
        image = Image.open(image_path).convert("RGB")
        print("Image Loaded Successfully")
    except Exception:
        raise FileNotFoundError("No Image found")
    
    # Setting the Text Labels
    text_labels = [["windows"]]

    # Feeding the Inputs to the Model
    inputs = processor(images = image, text = text_labels, return_tensors = "pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Generating the results
    results = processor.post_process_grounded_object_detection(outputs, threshold = threshold, target_sizes = [(image.height, image.width)])

    # Extracting the first result
    result = results[0]

    # Getting the number of detections
    detected_window_count = len(result["boxes"])
    print(f"Total {detected_window_count} windows are detected using threshold {threshold}")

    # Fallbacks if no window detected
    if detected_window_count == 0:
        print("No windows detected. Exiting.")
        return

    # Drawing the Detections on that Image
    draw = ImageDraw.Draw(image)

    # Store areas for all windows
    total_area_feet = 0.0

    # Compute pixel height of the first detected window
    first_box = result["boxes"][0].tolist()
    first_height_pixels = abs(first_box[3] - first_box[1])
    scale_factor = real_height_feet / first_height_pixels

    # Debug statements
    print(f"[INFO] Scaling factor calculated using first window:")
    print(f"[INFO] First window height (pixels): {first_height_pixels:.2f}")
    print(f"[INFO] Known real height (feet): {real_height_feet}")
    print(f"[INFO] Scale factor: {scale_factor:.5f} feet per pixel")

    # Process all detected boxes
    for i, (box, score, labels) in enumerate(zip(result["boxes"], result["scores"], result["labels"]), 1):
        box = [float(x) for x in box.tolist()]
        label = labels 
        confidence = float(score.item())

        print(f"Detected {label} with confidence {confidence} at location {box}")

        # Calculating the pixel wise height and width of each box
        x_min, y_min, x_max, y_max = box
        width_px = abs(x_max - x_min)
        height_px = abs(y_max - y_min)

        # Convert pixel unit into real world units
        width_ft = width_px * scale_factor
        height_ft = height_px * scale_factor
        area_ft = width_ft * height_ft
        total_area_feet += area_ft

        # Debug statements
        print(f"Window {i}:")
        print(f"Real Width x Real Height (feet): {width_ft:.2f} x {height_ft:.2f}")
        print(f"Area: {area_ft:.2f} sq. feet")

        # Draw bounding box: The box coordinates are [x_min, y_min, x_max, y_max]
        draw.rectangle(box, outline = "red", width = 3)

        # Draw label and confidence: Positioning text slightly above the box, or inside if too close to top
        text_position = (x_min, y_min - 25 if y_min > 25 else y_min + 5)
        draw.text(text_position, f"{label} {confidence:.2f}", fill = "red")

    # Print the Total Objects area
    print(f"\n[RESULT] Total window area = {total_area_feet:.2f} sq. feet")

    # PIL to Numpy conversion
    image_np = np.array(image)
    
    # Put the Total Area of the objects on the image
    cv2.putText(image_np, f"Total Object Area: {np.round(total_area_feet,0)} Sq. Feet", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Convert back to PIL image so that image.show() works
    image = Image.fromarray(image_np)
    
    # Save the image with detections
    if output_path is None:
        pass
    else:
        image.save(output_path)
        print(f"Image with detections saved to: {output_path}")

    # Display the image (optional, as it might block execution in some environments)
    image.show()

    # Return the Total objects area
    return np.round(total_area_feet, 0)

# Inference on the Image_8.jpg
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_7.jpg"
    output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_7_Objects_Area.jpg"
    object_area = detect_objects(image_path, real_height_feet = 3.5, threshold = 0.3, output_path = output_path)
    print("The Total Object Area is:", object_area)


############################# FOR MULTIPLE OBJECTS ###############################

# # Importing Required Packages
# import torch
# from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
# from PIL import Image, ImageDraw
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
# import cv2

# # Setting the Device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("[DEBUG] The device is:", device)

# # Prepare processor and model
# model_id = "rziga/mm_grounding_dino_large_all"
# print("[DEBUG] Setting the Model Name")

# processor = AutoProcessor.from_pretrained(model_id)
# print("[DEBUG] The processor is Loaded Successfully")

# model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
# print("[DEBUG] The Model is Loaded Successfully")

# # Define Function to get different detections
# def detect_objects(image_path: str, real_height_window_feet: float = 3.5, real_height_door_feet: float = 7.0, threshold: float = 0.4, output_path = None):

#     """
#     Detects windows and doors, computes both per-object and total real-world areas (sq.ft),
#     and annotates them on the output image.

#     Args:
#         image_path (str): Path to the input image.
#         real_height_window_feet (float): Real-world height of one window in feet.
#         real_height_door_feet (float): Real-world height of one door in feet.
#         threshold (float): Confidence threshold for detection.
#         output_path (str, optional): Path to save the annotated image.
#     """

#     # --- Load Image ---
#     try:
#         image = Image.open(image_path).convert("RGB")
#         print("Image Loaded Successfully")
#     except Exception:
#         raise FileNotFoundError("No Image found at:", image_path)

#     # --- Prepare Labels ---
#     text_labels = [["windows", "doors"]]
#     print("[DEBUG] Using text labels:", text_labels)

#     # --- Model Input ---
#     inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     # --- Postprocess Detections ---
#     results = processor.post_process_grounded_object_detection(
#         outputs, threshold=threshold, target_sizes=[(image.height, image.width)]
#     )[0]

#     detected_count = len(results["boxes"])
#     if detected_count == 0:
#         print("No objects detected. Exiting.")
#         return

#     print(f"Total {detected_count} objects detected at threshold {threshold}")

#     draw = ImageDraw.Draw(image)

#     # --- Initialize Accumulators ---
#     total_area_all = 0.0
#     object_areas = {"windows": 0.0, "doors": 0.0}
#     scale_factors = {"windows": None, "doors": None}
#     real_heights = {"windows": real_height_window_feet, "doors": real_height_door_feet}

#     # --- Process Each Detection ---
#     for i, (box, score, label_id) in enumerate(zip(results["boxes"], results["scores"], results["labels"]), 1):
#         box = [float(x) for x in box.tolist()]
#         label_text = label_id if isinstance(label_id, str) else text_labels[0][label_id] # text_labels[0][label_id] or label_text = label_id
#         confidence = float(score.item())

#         print(f"\nDetected {label_text} with confidence {confidence:.2f} at {box}")

#         # Pixel dimensions
#         x_min, y_min, x_max, y_max = box
#         width_px = abs(x_max - x_min)
#         height_px = abs(y_max - y_min)

#         # Compute scale factor once per object type
#         if scale_factors[label_text] is None:
#             scale_factor = real_heights[label_text] / height_px
#             scale_factors[label_text] = scale_factor
#             print(f"[INFO] Scale factor for {label_text}: {scale_factor:.5f} ft/pixel")

#         scale_factor = scale_factors[label_text]
#         width_ft = width_px * scale_factor
#         height_ft = height_px * scale_factor
#         area_ft = width_ft * height_ft

#         object_areas[label_text] += area_ft
#         total_area_all += area_ft

#         print(f"{label_text.capitalize()} {i}: {width_ft:.2f} ft x {height_ft:.2f} ft = {area_ft:.2f} sq.ft")

#         # Draw bounding boxes and labels
#         color = (255, 0, 0) if label_text == "windows" else (0, 0, 255)
#         draw.rectangle(box, outline="red" if label_text == "windows" else "blue", width=3)
#         text_position = (x_min, y_min - 25 if y_min > 25 else y_min + 5)
#         draw.text(text_position, f"{label_text} {confidence:.2f}",
#                   fill="red" if label_text == "windows" else "blue")

#     # --- Print Summary ---
#     print("\n[RESULT] Object-wise total areas (sq.ft):")
#     for obj, area in object_areas.items():
#         print(f"  {obj}: {area:.2f}")
#     print(f"[RESULT] Combined Total Area: {total_area_all:.2f} sq.ft")

#     # --- Annotate Summary on Image ---
#     image_np = np.array(image)
#     y_offset = 40
#     for obj, area in object_areas.items():
#         cv2.putText(image_np, f"{obj.capitalize()} Area: {area:.2f} sq.ft",
#                     (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
#         y_offset += 30

#     cv2.putText(image_np, f"Total Area: {total_area_all:.2f} sq.ft",
#                 (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

#     image = Image.fromarray(image_np)

#     # --- Save & Display ---
#     if output_path:
#         image.save(output_path)
#         print(f"Image with detections saved to: {output_path}")

#     image.show()

#     return {**object_areas, "total_area": total_area_all}


# # Inference on the Modified Function
# if __name__ == "__main__":
#     image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_3.jpg"
#     output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_3_Modified_Objects_Area.jpg"

#     areas = detect_objects(image_path = image_path, real_height_window_feet = 3.5, real_height_door_feet = 7.0, threshold = 0.3, output_path = output_path)

#     print("\nFinal Area Summary:")
#     for obj, area in areas.items():
#         print(f"{obj}: {area:.2f} sq.ft")