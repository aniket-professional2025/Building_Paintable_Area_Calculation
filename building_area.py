# # Importing required packages
# import numpy as np
# import cv2

# # The function to count floors based on Visible Lines
# def building_information(image_path, tolerence, actual_floor_height = 10, output_path = None):

#     # Load the image
#     try:
#         img = cv2.imread(image_path)
#     except Exception as e:
#         # Check if the image path is a string first, then handle file not found
#         if not isinstance(image_path, str):
#             raise TypeError("Image path must be a string.")
#         # Check if the loaded image is None, which happens if the path is invalid
#         if img is None:
#              raise FileNotFoundError(f"Image not found at path: {image_path}")
    
#     # Convert the image into gray scale image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Applying Gaussian Blur
#     blur = cv2.GaussianBlur(gray, (5,5), 0)

#     # Finding the Edges
#     edges = cv2.Canny(blur, 50, 150)

#     # Detect Lines using Hough Transform
#     # FIX: Corrected 'hough_threshold' to 'threshold' for cv2.HoughLinesP
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 100, minLineLength = 100, maxLineGap = 10)

#     # What if no lines are detected
#     if lines is None:
#         print("No lines detected.")
#         return [], 0, 0, 0 # Added return of 0 for height and width
    
#     # Creating the horizontal Lines and finding the building's pixel span
#     horizontal_lines = []
#     min_x = float('inf')
#     max_x = float('-inf')

#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         # Only consider lines that are nearly horizontal (slope close to zero)
#         if abs(y2 - y1) < 10: 
#             horizontal_lines.append((y1 + y2) // 2)
            
#             # Find the minimum and maximum x-coordinates across ALL relevant horizontal lines
#             min_x = min(min_x, x1, x2)
#             max_x = max(max_x, x1, x2)

#     # Estimate building width in pixels as the total horizontal span
#     building_width_pixels = max_x - min_x if max_x > min_x else 0
    
#     # Cluster Similar Lines
#     horizontal_lines = sorted(horizontal_lines)
#     floor_lines = []
#     for y in horizontal_lines:
#         if not floor_lines or abs(y - floor_lines[-1]) > tolerence:
#             floor_lines.append(y)
    
#     # Count floors
#     num_floors = len(floor_lines)

#     # Compute distances (in pixels) between floor lines
#     floor_distances = []
#     for i in range(1, len(floor_lines)):
#         dist = abs(floor_lines[i] - floor_lines[i - 1])
#         floor_distances.append(dist)

#     # Assuming each floor is 10 feet tall
#     scale_factor = None
#     if floor_distances:
#         avg_floor_pixel_height = np.mean(floor_distances)
#         scale_factor = actual_floor_height / avg_floor_pixel_height   # feet per pixel
    
#     # Find the height and width of the building in feet unit
#     real_height = num_floors * 10 
    
#     # Width calculation uses the total pixel span found earlier
#     if scale_factor is not None and building_width_pixels > 0:
#         real_width = scale_factor * building_width_pixels
#     else:
#         real_width = 0

#     # Calculate the Area of the Building
#     area = real_width * real_height
#     area = np.round(area, 2)

#     # Annotate the Image: 
#     for y in floor_lines:
#         cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 2)

#     # Put the Answer on the image
#     cv2.putText(img, f"Estimated Floors: {num_floors}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(img, f"Estimated Height : {real_height} Feets", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,  0.7, (0,0,255), 2, cv2.LINE_AA)
#     cv2.putText(img, f"Estimated Width : {np.round(real_width,2)} Feets", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(img, f"Estimated Area : {np.round(area, 2)} Sq.Feets", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

#     # Saving the Image in Output Path
#     if output_path is None:
#         pass
#     else:
#         cv2.imwrite(output_path, img, params = None)

#     cv2.imshow("Detected Floors", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Return the result
#     return num_floors, real_height, np.round(real_width, 2), area

# # The Inference Code
# if __name__ == "__main__":
#     image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_8.jpg"
#     output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\images\Image_8_Test.jpg"
    
#     # hough_threshold was increased to 100 in your latest script, keeping it that way
#     floor_num, height, width, area = building_information(image_path, tolerence = 130, output_path = output_path)
    
#     print("The number of Floors in the image is:", floor_num)
#     print("The Actual Height in Feet is:", height)
#     print("The Actual Width in Feet is:", width)
#     print("The Area of the Building in Sq. Feet is:", area)

###################################################################################################

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

# # Function to detect windows using image_path and threshold
# def detect_objects(image_path: str, detection_threshold: float, real_building_height: int = 40, output_path = None):

#     # Reading the Image
#     try:
#         image = Image.open(image_path).convert("RGB")
#         print("[DEBUG] Inside the Function: Image Loaded Successfully")
#     except Exception:
#         raise FileNotFoundError("No Image found")
    
#     # Setting the Text Labels
#     text_labels = [["building"]]

#     # Feeding the Inputs to the Model
#     inputs = processor(images = image, text = text_labels, return_tensors = "pt").to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Generating the results
#     results = processor.post_process_grounded_object_detection(outputs, threshold = detection_threshold, target_sizes = [(image.height, image.width)])

#     # Extracting the first result
#     result = results[0]
#     print("[DEBUG] The results are Achieved")

#     # Getting the number of detections
#     detections = len(result["boxes"])
#     print(f"[DEBUG] Total {detections} buildings are detected using threshold {detection_threshold}")

#     # Fallbacks if no window detected
#     if detections == 0:
#         print("[DEBUG] No Building detected. Exiting the Function")
#         return

#     # Drawing the Detections on that Image
#     draw = ImageDraw.Draw(image)

#     # Get the box coordinate and its height and width
#     box = result['boxes'][0].tolist()
#     height_px = abs(box[3] - box[1])
#     width_px = abs(box[2] - box[0])

#     # The detected box on the image
#     draw.rectangle(box, outline = 'red', width = 3)
    
#     # Calculate the scale factor
#     scale_factor = real_building_height / height_px

#     # Calculate the Actual Height and Actual Width
#     height_ft = real_building_height
#     width_ft = np.round(scale_factor * width_px, 0)

#     # Calculate the Area of the building in feet
#     area_ft = np.round(height_ft * width_ft, 0)

#     # Putting Text on the Image
#     image_np = np.array(image)
#     cv2.putText(image_np, f"Height : {height_ft} Ft", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(image_np, f"Width : {width_ft} Ft", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(image_np, f"Area : {area_ft} Sq. Ft", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

#     # Printing the results
#     print("The bounding Box Coordinates are:", result['boxes'].tolist()[0])
#     print(f"The Height and Width of the Buidling are: {height_ft} feet and {width_ft} feet")
#     print(f"The Area of the Building is: {area_ft} sq. feet")

#     # Converting the Image back in PIL format and showing it
#     image = Image.fromarray(image_np)

#     # Saving the image in the specified output path
#     if output_path is None:
#         pass
#     else: 
#         image.save(output_path)
#         print(f"The Annotated Image is Saved in: {output_path}")

#     # Showing the Image
#     image.show()

# # Inference
# image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\OrgImages\Image_7.jpg"
# output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\Modify\Image_7_Result.jpg"
# detect_objects(image_path, detection_threshold = 0.5, real_building_height = 40, output_path = output_path)