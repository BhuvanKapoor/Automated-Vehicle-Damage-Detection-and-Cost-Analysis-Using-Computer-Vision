import cv2
import numpy as np
from ultralytics import YOLO

def detect_car_parts(image):
    # model = YOLO("models\carpartdetection_best.pt")
    model = YOLO("models\\best.pt")
    results = model.predict(image, save=False, show=False)

    class_names = ['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield']
    color_map = {
        "Bumper": (255, 0, 0), "Door": (0, 255, 0), "Fender": (0, 0, 255),"Bonnet": (255, 165, 0), "Windshield": (128, 0, 128), "Dickey": (255, 140, 0)
    }
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    part_centers = []
    
    for result in results:
        boxes = result.boxes
        class_ids = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        for box, cls_id, score in zip(boxes.xyxy, class_ids, scores):
            part_name = class_names[int(cls_id)]
            color = color_map.get(part_name, (255, 255, 255))
            
            if score > 0.25:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{part_name} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                part_centers.append((center_x, center_y, part_name))
    
    return image, part_centers
