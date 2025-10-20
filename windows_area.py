# Importing Required Packages
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw, ImageFont 
import numpy as np

# Prepare processor and model
model_id = "rziga/mm_grounding_dino_large_all"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[DEBUG] The device is:", device)

processor = AutoProcessor.from_pretrained(model_id)
print("[DEBUG] The processor is Loaded Successfully")

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("[DEBUG] The Model is Loaded Successfully")

# Function to detect windows using image_path and threshold
def detect_windows(image_path: str, output_path: str, threshold: float = 0.40):
    image = Image.open(image_path).convert("RGB")
    text_labels = [["windows"]]
    inputs = processor(images = image, text = text_labels, return_tensors = "pt").to(device)


    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(outputs, threshold = threshold, target_sizes = [(image.height, image.width)])

    result = results[0]

    detected_window_count = len(result["boxes"])
    print(f"Total {detected_window_count} windows are detected using {threshold} threshold")

    draw = ImageDraw.Draw(image)

    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        label = labels
        confidence = round(score.item(), 3)

        print(f"Detected {label} with confidence {confidence} at location {box}")

    # Draw bounding box: The box coordinates are [x_min, y_min, x_max, y_max]
    draw.rectangle(box, outline = "red", width = 3)

    # Draw label and confidence: Positioning text slightly above the box, or inside if too close to top
    text_position = (box[0], box[1] - 25 if box[1] > 25 else box[1] + 5)
    draw.text(text_position, f"{label} {confidence}", fill = "red")

    # Save the image with detections
    output_image_path = output_path
    image.save(output_image_path)
    print(f"\nImage with detections saved to: {output_image_path}")

    # Display the image (optional, as it might block execution in some environments)
    image.show() 


# Inference on the Image_7
image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_7.jpg"
output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_7_Windows.jpg"
detect_windows(image_path, output_path)