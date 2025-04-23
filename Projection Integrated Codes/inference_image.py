
import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os
import json

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

def parse_arguments(scene_directory):
    """Parse command line arguments."""
    # scene_directory = r"C:\Users\farha\OneDrive\Desktop\Lanes_integrated\input\curvedlanesdata\scene3"  # input scene 
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', 
        '--input', 
        default=scene_directory,
        required=False, 
        help='path to the input data'
    )
    parser.add_argument(
        '-t', 
        '--threshold', 
        default=0.8, 
        type=float,
        help='score threshold for discarding detection'
    )
    parser.add_argument(
        '-w',
        '--weights',
        default=r'C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\outputs\training\road_line\model_15.pth',
        help='path to the trained wieght file'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to visualize the results in real-time on screen'
    )
    parser.add_argument(
        '--no-boxes',
        action='store_true',
        default=True,
        help='do not show bounding boxes, only show segmentation map'
    )
    parser.add_argument(
        '--outdir',
        default='current',
        help='output directory for saving the images'
    )
    return parser.parse_args()

def determine_mask_color(image, mask, box):
    """Determine if a mask is yellow or white based on HSV values."""
    # Convert PIL Image to numpy array for OpenCV
    image_np = np.array(image)
    # Convert to BGR (OpenCV format)
    image_bgr = image_np[:, :, ::-1].copy() if image_np.shape[2] == 3 else image_np
    # Convert to HSV color space
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Get box coordinates (top-left, bottom-right)
    (x1, y1), (x2, y2) = box
    
    # Ensure coordinates are within image bounds
    height, width = image_hsv.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width-1, x2), min(height-1, y2)
    
    # Extract region of interest from the mask and image
    roi_mask = mask[y1:y2, x1:x2]
    roi_hsv = image_hsv[y1:y2, x1:x2]
    
    # Get pixels inside the mask
    mask_pixels = roi_hsv[roi_mask]
    
    # Get pixels inside the box but outside the mask
    inverse_mask = ~roi_mask
    non_mask_pixels = roi_hsv[inverse_mask]
    
    # Calculate average and maximum HSV values for both regions
    if len(mask_pixels) > 0:
        avg_mask_hsv = np.mean(mask_pixels, axis=0)
        max_mask_hsv = np.percentile(mask_pixels, 95, axis=0)  # 95th percentile to avoid outliers
        
        # Get percentile values for more detailed analysis
        p75_mask_s = np.percentile(mask_pixels[:, 1], 75)  # 75th percentile of saturation
        p90_mask_s = np.percentile(mask_pixels[:, 1], 90)  # 90th percentile of saturation
        
        # Histogram of hue values to identify dominant colors
        hue_hist, _ = np.histogram(mask_pixels[:, 0], bins=36, range=(0, 180))
        dominant_hue_bin = np.argmax(hue_hist)
        dominant_hue = dominant_hue_bin * 5  # Convert bin index to hue value (180/36 = 5)
    else:
        return "unknown", {}, {}  # No mask pixels
    
    if len(non_mask_pixels) > 0:
        avg_non_mask_hsv = np.mean(non_mask_pixels, axis=0)
        max_non_mask_hsv = np.percentile(non_mask_pixels, 95, axis=0)
    else:
        avg_non_mask_hsv = np.array([0, 0, 0])
        max_non_mask_hsv = np.array([0, 0, 0])
    
    # Prepare detailed HSV stats
    mask_hsv_stats = {
        "avg": avg_mask_hsv.tolist(),
        "max": max_mask_hsv.tolist(),
        "p75_saturation": float(p75_mask_s),
        "p90_saturation": float(p90_mask_s),
        "dominant_hue": float(dominant_hue)
    }
    
    non_mask_hsv_stats = {
        "avg": avg_non_mask_hsv.tolist(),
        "max": max_non_mask_hsv.tolist()
    }
    
    # Yellow detection criteria using max values and percentiles
    # Yellow typically has hue around 20-40
    is_yellow_hue = 15 <= dominant_hue <= 45
    is_high_saturation = p90_mask_s >= 100
    
    # White detection criteria
    is_low_saturation = p90_mask_s < 60
    is_high_value = avg_mask_hsv[2] > 150
    
    if is_yellow_hue and is_high_saturation:
        color = "yellow"
    elif is_low_saturation and is_high_value:
        color = "white"
    else:
        color = "other"
    
    return color, mask_hsv_stats, non_mask_hsv_stats

def load_model(weights_path, device):
    """Load and initialize the model."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        pretrained=False, num_classes=91
    )

    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
    model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    # set to eval mode
    model.to(device).eval()
    
    return model

def setup_transforms():
    """Setup image transformations."""
    return transforms.Compose([
        transforms.ToTensor()
    ])

def setup_directories(args):
    """Setup output directories."""
    out_dir = os.path.join('outputs', args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    
    scene_dir = 'current_scene_masks'
    os.makedirs(scene_dir, exist_ok=True)
    
    return out_dir, scene_dir

def process_image(image_path, model, transform, device, args, out_dir, scene_dir):
    """Process a single image."""
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    
    # transform the image
    image_tensor = transform(image)
    # add a batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    masks, boxes, labels = get_outputs(image_tensor, model, args.threshold)
    
    result, mask_coords = draw_segmentation_map(orig_image, masks, boxes, labels, args)
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # Create image-specific directory under scene
    image_dir = os.path.join(scene_dir, image_name)
    os.makedirs(image_dir, exist_ok=True)
    
    # Process each mask and determine color
    color_info = {}
    
    for mask_idx, (mask, box) in enumerate(zip(masks, boxes)):
        # Determine if the mask is yellow or white
        try:
            color, mask_hsv_stats, non_mask_hsv_stats = determine_mask_color(orig_image, mask, box)
            
            # Save information
            color_info[f"mask_{mask_idx}"] = {
                "color": color,
                "mask_hsv_stats": mask_hsv_stats,
                "non_mask_hsv_stats": non_mask_hsv_stats,
                "box_coordinates": [[box[0][0], box[0][1]], [box[1][0], box[1][1]]]  # Convert tuples to lists for JSON
            }
        except Exception as e:
            print(f"Error processing mask {mask_idx}: {e}")
            color_info[f"mask_{mask_idx}"] = {
                "color": "error",
                "error": str(e),
                "box_coordinates": [[box[0][0], box[0][1]], [box[1][0], box[1][1]]] if len(box) == 2 else "unknown format"
            }
        
        # Save coordinates for each individual mask
        with open(os.path.join(image_dir, f'mask_{mask_idx}_coordinates.txt'), 'w') as f:
            for coord in mask_coords[mask_idx]:
                f.write(f"{coord[0]},{coord[1]}\n")
    
    # Save color information in a JSON file
    with open(os.path.join(image_dir, 'color_info.json'), 'w') as f:
        json.dump(color_info, f, indent=4)
    
    # Save the visualization
    save_path = f"{out_dir}/{os.path.basename(image_path).split('.')[0]}.jpg"
    cv2.imwrite(save_path, result)
    
    print(f"Processed image: Found {len(masks)} masks")
    for mask_idx, info in color_info.items():
        if info.get("color") != "error":
            # Get the averages for display
            mask_avg = info['mask_hsv_stats']['avg'] if 'avg' in info['mask_hsv_stats'] else [0, 0, 0]
            dominant_hue = info['mask_hsv_stats'].get('dominant_hue', 0)
            p90_sat = info['mask_hsv_stats'].get('p90_saturation', 0)
            
            print(f"  {mask_idx}: {info['color']} (dominant hue: {dominant_hue:.1f}, p90 saturation: {p90_sat:.1f}, avg HSV: {mask_avg})")
        else:
            print(f"  {mask_idx}: Error processing - {info.get('error')}")
    
    return len(masks)

def execute_masks(scene_directory):
    """Main function to execute the program."""
    # Parse arguments
    args = parse_arguments(scene_directory)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.weights, device)
    
    # Setup transforms
    transform = setup_transforms()
    
    # Setup directories
    out_dir, scene_dir = setup_directories(args)
    
    # Get image paths
    image_paths = glob.glob(os.path.join(args.input, '*.jpg'))
    
    # Process images
    total_masks = 0
    for num, image_path in enumerate(image_paths):
        masks_count = process_image(
            image_path, model, transform, device, args, out_dir, scene_dir
        )
        total_masks += masks_count
        print(f"Processed image {num+1}/{len(image_paths)}")
    
    print(f"Processing complete. Total images: {len(image_paths)}, Total masks: {total_masks}")


def main():
    """Main entry point."""
    execute_masks()
if __name__ == "__main__":
    main()