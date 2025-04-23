import cv2
from ultralytics import YOLO
import numpy as np

def extract_traffic_light_roi(img, coords):
    """Extracts a traffic light region of interest (ROI) using coordinates."""
    x1, y1, x2, y2 = coords
    
    # Add padding
    padding = 5
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.shape[1], x2 + padding)
    y2 = min(img.shape[0], y2 + padding)
    
    roi = img[y1:y2, x1:x2]
    return roi

def is_back_side(roi, sat_thresh=70, bright_thresh=75):
    """
    Improved back-side detection based on saturation and brightness.
    
    Args:
        roi (numpy.ndarray): Traffic light region of interest.
        sat_thresh (int): Saturation threshold for dull colors.
        bright_thresh (int): Brightness threshold for dark back sides.

    Returns:
        bool: True if it's likely the back side, False otherwise.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate mean saturation and brightness
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_brightness = np.mean(hsv[:, :, 2])
    
    # Check for back side characteristics
    if mean_saturation < sat_thresh and mean_brightness < bright_thresh:
        return True
    return False

def process_traffic_light_roi(roi):
    """Detects the traffic light color based on brightness sections."""
    height, width = roi.shape[:2]

    # Crop center 50%
    center_roi = roi[height//4:3*height//4, width//4:3*width//4]
    
    hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
    
    # Brightness (V channel)
    value_channel = hsv[:, :, 2]
    section_height = value_channel.shape[0] // 3

    sections = [
        value_channel[0:section_height, :],
        value_channel[section_height:2*section_height, :],
        value_channel[2*section_height:, :]
    ]
    
    brightness_sums = [np.sum(section) for section in sections]
    max_brightness_index = np.argmax(brightness_sums)

    # Classify based on brightness
    if max_brightness_index == 0:
        return 'Red'
    elif max_brightness_index == 1:
        return 'Yellow'
    else:
        return 'Green'

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1, centroid_radius=5):
    # Include all classes (including traffic lights)
    if not classes:
        classes = list(range(len(chosen_model.names)))
    
    results = chosen_model.predict(img, classes=classes, conf=conf)
    
    # Get traffic light class index
    traffic_light_index = next((idx for idx, name in chosen_model.names.items() if name.lower() == 'traffic light'), None)
    
    # Create lists to store the required coordinates
    bottom_left_info = []
    bottom_mid_info = []
    
    for result in results:
        for box in result.boxes:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class label
            class_id = int(box.cls[0])
            base_label = f"{result.names[class_id]}"
            
            # If it's a traffic light, determine its color
            if class_id == traffic_light_index:
                # Extract ROI for this traffic light
                traffic_light_roi = extract_traffic_light_roi(img, (x1, y1, x2, y2))
                
                # Check if it's a back side
                if not is_back_side(traffic_light_roi):
                    # Process the color
                    color = process_traffic_light_roi(traffic_light_roi)
                    # Modify label to include color
                    label = f"traffic_light_{color.lower()}"
                else:
                    # It's a back side, keep original label
                    label = base_label
            else:
                # For non-traffic light objects, use original label
                label = base_label
            
            # Draw bounding rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
            
            # Calculate bottom left corner (x1, y2)
            bottom_left_x = x1
            bottom_left_y = y2
            
            # Calculate bottom midpoint ((x1+x2)/2, y2)
            bottom_mid_x = (x1 + x2) // 2
            bottom_mid_y = y2
            
            # Draw bottom left point as a red circle
            # cv2.circle(img, (bottom_left_x, bottom_left_y), centroid_radius, (0, 0, 255), -1) 
            
            # Draw bottom midpoint as a green circle
            # cv2.circle(img, (bottom_mid_x, bottom_mid_y), centroid_radius, (0, 255, 0), -1)  
            
            # Display label
            cv2.putText(img, label, 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
            
            # Store information for text files
            bottom_left_info.append({
                "label": label,
                "coordinates": [bottom_left_x, bottom_left_y]
            })
            
            bottom_mid_info.append({
                "label": label,
                "coordinates": [bottom_mid_x, bottom_mid_y]
            })
    
    return img, results, bottom_left_info, bottom_mid_info

def main():
    # Load YOLO model
    model = YOLO("yolov9c.pt")
    
    # Read the image
    image = cv2.imread(r"C:\Users\farha\OneDrive\Desktop\ObjectDetection\Images2render\scene13\0034.jpg")
    
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Process the image and detect objects with traffic light colors
    result_img, detected_results, bottom_left_info, bottom_mid_info = predict_and_detect(model, image, conf=0.5)
    
    # Save bottom left corner coordinates to a text file
    # with open("bottom_left_corners_830.txt", "w") as f:

    #     for detection in bottom_left_info:
    #         f.write(f"{detection['label']}, {detection['coordinates'][0]}, {detection['coordinates'][1]}\n")
    
    # # Save bottom midpoint coordinates to a text file
    with open("bottom_midpoints_830.txt", "w") as f:

        for detection in bottom_mid_info:
            f.write(f"{detection['label']}, {detection['coordinates'][0]}, {detection['coordinates'][1]}\n")
    
    # Display and save the result
    cv2.imshow("Image with Points", result_img)
    # cv2.imwrite("scene4_8.jpg", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Bottom left corners saved to bottom_left_corners.txt")
    print(f"Bottom midpoints saved to bottom_midpoints.txt")

if __name__ == "__main__":
    main()