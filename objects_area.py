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

# Define Function to get different detections
def detect_objects(image_path: str, real_height_window_feet: float = 3.5, real_height_door_feet: float = 7.0, threshold: float = 0.4, output_path = None):

    # --- Load Image ---
    try:
        image = Image.open(image_path).convert("RGB")
        print("Image Loaded Successfully")
    except Exception:
        raise FileNotFoundError("No Image found at:", image_path)

    # --- Prepare Labels ---
    text_labels = [["windows", "doors"]]
    print("[DEBUG] Using text labels:", text_labels)

    # --- Model Input ---
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # --- Postprocess Detections ---
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=threshold, target_sizes=[(image.height, image.width)]
    )[0]

    detected_count = len(results["boxes"])
    if detected_count == 0:
        print("No objects detected. Exiting.")
        return

    print(f"Total {detected_count} objects detected at threshold {threshold}")

    draw = ImageDraw.Draw(image)

    # --- Initialize Accumulators ---
    total_area_all = 0.0
    object_areas = {"windows": 0.0, "doors": 0.0}
    scale_factors = {"windows": None, "doors": None}
    real_heights = {"windows": real_height_window_feet, "doors": real_height_door_feet}

    # --- Process Each Detection ---
    for i, (box, score, label_id) in enumerate(zip(results["boxes"], results["scores"], results["labels"]), 1):
        box = [float(x) for x in box.tolist()]
        label_text = label_id if isinstance(label_id, str) else text_labels[0][label_id] # text_labels[0][label_id] or label_text = label_id
        confidence = float(score.item())

        print(f"\nDetected {label_text} with confidence {confidence:.2f} at {box}")

        # Pixel dimensions
        x_min, y_min, x_max, y_max = box
        width_px = abs(x_max - x_min)
        height_px = abs(y_max - y_min)

        # Compute scale factor once per object type
        if scale_factors[label_text] is None:
            scale_factor = real_heights[label_text] / height_px
            scale_factors[label_text] = scale_factor
            print(f"[INFO] Scale factor for {label_text}: {scale_factor:.5f} ft/pixel")

        scale_factor = scale_factors[label_text]
        width_ft = np.round(width_px * scale_factor, 0)
        height_ft = np.round(height_px * scale_factor, 0)
        area_ft = np.round(width_ft * height_ft, 0)

        object_areas[label_text] += area_ft
        total_area_all += area_ft

        print(f"{label_text.capitalize()} {i}: {width_ft:.2f} ft x {height_ft:.2f} ft = {area_ft:.2f} sq.ft")

        # Draw bounding boxes and labels
        color = (255, 0, 0) if label_text == "windows" else (0, 0, 255)
        draw.rectangle(box, outline="red" if label_text == "windows" else "blue", width=3)
        text_position = (x_min, y_min - 25 if y_min > 25 else y_min + 5)
        draw.text(text_position, f"{label_text} {confidence:.2f}",
                  fill="red" if label_text == "windows" else "blue")

    # --- Print Summary ---
    print("\n[RESULT] Object-wise total areas (sq.ft):")
    for obj, area in object_areas.items():
        print(f"  {obj}: {area:.2f}")
    print(f"[RESULT] Combined Total Area: {total_area_all:.2f} sq.ft")

    # --- Annotate Summary on Image ---
    image_np = np.array(image)
    y_offset = 40
    for obj, area in object_areas.items():
        cv2.putText(image_np, f"{obj.capitalize()} Area: {area:.2f} sq.ft",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        y_offset += 30

    cv2.putText(image_np, f"Total Area: {total_area_all:.2f} sq.ft",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    image = Image.fromarray(image_np)

    # --- Save & Display ---
    if output_path:
        image.save(output_path)
        print(f"Image with detections saved to: {output_path}")

    image.show()

    # Creating the Json for storing the Final result
    result_json =  {**object_areas, "total_area": total_area_all}

    # Accessing the area of each object
    window_area = result_json['windows']
    door_area = result_json['doors']
    total_area = result_json['total_area']

    # Returning the areas
    return window_area, door_area, total_area

# # Inference on the Modified Function
# if __name__ == "__main__":
#     image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\OrgImages\Image_3.jpg"
#     output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\Initial_Results\Image_3_Modified_Objects_Area.jpg"

#     window, door, total = detect_objects(image_path = image_path, real_height_window_feet = 3.5, real_height_door_feet = 7.0, threshold = 0.3, output_path = output_path)

#     print("Window Area:", window)
#     print("Door Area:", door)
#     print("Total Area:", total)