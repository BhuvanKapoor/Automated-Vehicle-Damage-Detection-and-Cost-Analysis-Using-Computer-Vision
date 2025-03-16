import cv2
import numpy as np
from ultralytics import YOLO

def generate_damage_mask(image):
    # Load YOLO model
    model = YOLO("models\cardamageseg_best.pt")

    # Perform prediction
    results = model.predict(image, save=False, show=False)

    # Define class names and assign distinct colors
    class_names = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
    color_map = {
        "dent": (0, 255, 0),         # Green
        "scratch": (255, 0, 0),      # Blue
        "crack": (0, 0, 255),        # Red
        "glass shatter": (255, 255, 0),  # Cyan
        "lamp broken": (255, 165, 0),    # Orange
        "tire flat": (128, 0, 128)       # Purple
    }

    # Convert PIL Image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    damage_centers = []

    # Iterate over predictions
    for result in results:
        masks = result.masks  # Segmentation masks
        boxes = result.boxes  # Bounding boxes
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs (converted to numpy)
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores

        for mask, box, cls_id, score in zip(masks.xy, boxes.xyxy, class_ids, scores):
            if score > 0.45:
                # Convert mask points to integer
                mask_points = np.array(mask, dtype=np.int32)

                # Get damage type name
                damage_type = class_names[int(cls_id)]

                # Get corresponding color
                color = color_map.get(damage_type, (255, 255, 255))  # Default to white if unknown

                # Draw segmentation mask
                overlay = image.copy()
                cv2.fillPoly(overlay, [mask_points], color)  # Apply the color
                image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.cpu().numpy())  # Convert tensor to numpy
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                # Calculate center and radius
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) // 2
                damage_centers.append((center_x, center_y, radius, damage_type))

                # Put class label with the same color
                cv2.putText(image, f"{damage_type} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image, damage_centers