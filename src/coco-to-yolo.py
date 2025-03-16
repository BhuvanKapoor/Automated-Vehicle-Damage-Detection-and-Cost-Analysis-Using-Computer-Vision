import os
import json
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

def create_folder_structure(base_dir):
    """Create the YOLO folder structure."""
    # Main directories
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "test"), exist_ok=True)
    
    return base_dir

def convert_coco_to_yolo(input_dir, output_dir):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        input_dir (str): Path to the Cardd_coco directory
        output_dir (str): Path to save YOLO format dataset
    """
    # Create YOLO dataset directory structure
    yolo_dir = create_folder_structure(output_dir)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        
        # Load COCO annotations
        annotation_file = os.path.join(input_dir, 'annotations', f'instances_{split}.json')
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image ID to filename mapping
        image_id_to_filename = {}
        for image_info in coco_data['images']:
            image_id_to_filename[image_info['id']] = {
                'file_name': image_info['file_name'],
                'width': image_info['width'],
                'height': image_info['height']
            }
        
        # Create class ID mapping (COCO class ID to YOLO class ID)
        # YOLO expects classes to be numbered from 0
        coco_to_yolo_class = {}
        for i, category in enumerate(coco_data['categories']):
            coco_to_yolo_class[category['id']] = i
        
        # Save class names to dataset.yaml
        with open(os.path.join(yolo_dir, 'dataset.yaml'), 'w') as f:
            f.write(f"path: {os.path.abspath(yolo_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n\n")
            f.write("nc: " + str(len(coco_data['categories'])) + "\n")
            f.write("names: [")
            for i, category in enumerate(coco_data['categories']):
                if i > 0:
                    f.write(", ")
                f.write(f"'{category['name']}'")
            f.write("]\n")
        
        # Group annotations by image
        image_to_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_to_annotations:
                image_to_annotations[image_id] = []
            image_to_annotations[image_id].append(ann)
        
        # Process each image
        for image_id, annotations in tqdm(image_to_annotations.items()):
            if image_id not in image_id_to_filename:
                continue
                
            image_info = image_id_to_filename[image_id]
            img_width = image_info['width']
            img_height = image_info['height']
            
            # Copy image to YOLO directory
            src_img_path = os.path.join(input_dir, split, image_info['file_name'])
            dst_img_path = os.path.join(yolo_dir, 'images', split, image_info['file_name'])
            shutil.copy(src_img_path, dst_img_path)
            
            # Create YOLO format annotation file
            txt_filename = os.path.splitext(image_info['file_name'])[0] + '.txt'
            txt_path = os.path.join(yolo_dir, 'labels', split, txt_filename)
            
            with open(txt_path, 'w') as f:
                for ann in annotations:
                    # Get YOLO class ID
                    yolo_class_id = coco_to_yolo_class[ann['category_id']]
                    
                    # Process segmentation
                    if 'segmentation' in ann and len(ann['segmentation']) > 0:
                        # YOLO format for segmentation: class_id x1 y1 x2 y2 ... xn yn
                        for seg in ann['segmentation']:
                            # Normalize coordinates
                            points = []
                            for i in range(0, len(seg), 2):
                                if i+1 < len(seg):
                                    x = seg[i] / img_width
                                    y = seg[i+1] / img_height
                                    points.extend([x, y])
                            
                            # Write to file
                            if points:
                                f.write(f"{yolo_class_id} ")
                                f.write(" ".join([f"{p:.6f}" for p in points]))
                                f.write("\n")

def extract_masks_from_sod(input_dir, output_dir):
    """
    Extract masks from Cardd_sod and match them with YOLO format.
    
    Args:
        input_dir (str): Path to the Cardd_sod directory
        output_dir (str): Path to the YOLO format dataset
    """
    for split in ['train', 'val', 'test']:
        mask_dir = os.path.join(input_dir, f'CarDD-{split}', f'CarDD-{split}-Mask')
        img_dir = os.path.join(input_dir, f'CarDD-{split}', f'CarDD-{split}-Image')
        
        # Create mask directory in YOLO structure
        os.makedirs(os.path.join(output_dir, 'masks', split), exist_ok=True)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        for img_file in tqdm(image_files, desc=f"Processing {split} masks"):
            # Check if corresponding mask exists
            mask_file = img_file
            mask_path = os.path.join(mask_dir, mask_file)

            if os.path.exists(mask_path):
                # Copy mask to YOLO directory
                dst_mask_path = os.path.join(output_dir, 'masks', split, img_file)
                shutil.copy(mask_path, dst_mask_path)
                
                # Copy corresponding image if it doesn't exist already
                yolo_img_path = os.path.join(output_dir, 'images', split, img_file)
                src_img_path = os.path.join(img_dir, img_file)
                
                if not os.path.exists(yolo_img_path) and os.path.exists(src_img_path):
                    shutil.copy(src_img_path, yolo_img_path)

def main():
    # Dataset paths
    cardd_coco_dir = "Data\CarDD_dataset\CarDD_COCO"
    cardd_sod_dir = "Data\CarDD_dataset\CarDD_SOD"
    yolo_output_dir = "Data\yolov11_format"
    
    # Create YOLO format dataset directory
    os.makedirs(yolo_output_dir, exist_ok=True)
    
    # Convert COCO format to YOLO
    convert_coco_to_yolo(cardd_coco_dir, yolo_output_dir)
    
    # Extract and integrate mask data from SOD
    extract_masks_from_sod(cardd_sod_dir, yolo_output_dir)
    
    print(f"Conversion completed. YOLO format dataset saved to {yolo_output_dir}")
    
    # Generate data.yaml file with paths for YOLOv11
    with open(os.path.join(yolo_output_dir, "data.yaml"), "w") as f:
        f.write(f"train: {os.path.abspath(os.path.join(yolo_output_dir, 'images', 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(yolo_output_dir, 'images', 'val'))}\n")
        f.write(f"test: {os.path.abspath(os.path.join(yolo_output_dir, 'images', 'test'))}\n\n")
        
        # Get number of classes from dataset.yaml
        with open(os.path.join(yolo_output_dir, "dataset.yaml"), "r") as ds_yaml:
            for line in ds_yaml:
                if line.startswith("nc:"):
                    f.write(line)
                if line.startswith("names:"):
                    f.write(line)
        
        # Add mask paths if available
        f.write(f"\nmasks:\n")
        f.write(f"  train: {os.path.abspath(os.path.join(yolo_output_dir, 'masks', 'train'))}\n")
        f.write(f"  val: {os.path.abspath(os.path.join(yolo_output_dir, 'masks', 'val'))}\n")
        f.write(f"  test: {os.path.abspath(os.path.join(yolo_output_dir, 'masks', 'test'))}\n")

if __name__ == "__main__":
    main()