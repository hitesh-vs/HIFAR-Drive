import cv2
import numpy as np
import torch

from class_names import INSTANCE_CATEGORY_NAMES as coco_names

np.random.seed(2023)

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the model.
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels    

def process_multiple_images(image_paths, model, output_dir='lane_data'):
    """
    Process multiple images and extract lane pixel values for each
    
    Args:
        image_paths: List of paths to image files
        model: Your trained lane detection model
        output_dir: Directory to save lane data
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare for model and run inference
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        masks, boxes, labels = get_outputs(image_tensor, model, threshold=0.5)
        
        # Get filename without extension for saving
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Extract and save lane pixel values
        get_lane_pixel_values(image_rgb, masks, labels, frame_number=i, 
                             output_dir=os.path.join(output_dir, base_filename))

def get_lane_pixel_values(image, masks, labels, frame_number=0, output_dir='lane_pixel_data'):
    """
    Extract pixel values for lanes from the original image based on masks and save to NPY file.
    Each frame gets its own directory with separate files for each lane.
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert image to numpy array if it's a tensor
    if torch.is_tensor(image):
        # Check if we need to reshape
        if len(image.shape) == 3 and image.shape[0] == 3:  # (C, H, W) format
            image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
        elif len(image.shape) == 4 and image.shape[0] == 1:  # (B, C, H, W) format with batch=1
            image_np = image[0].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
        else:
            image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    # Make sure image is in the correct format (H, W, C) or (H, W)
    if len(image_np.shape) == 3 and image_np.shape[2] not in [1, 3]:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    lane_data = {}
    
    # Process each mask
    for i in range(len(masks)):
        print(f"Processing mask {i}, label: {labels[i]}")
        
        # Simplified label check - process all masks
        # Get mask as boolean array
        mask_bool = masks[i].astype(bool)
        
        # Check dimension compatibility
        if mask_bool.shape[:2] != image_np.shape[:2]:
            print(f"Dimension mismatch: mask {mask_bool.shape[:2]}, image {image_np.shape[:2]}")
            try:
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), 
                                      (image_np.shape[1], image_np.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            except Exception as e:
                print(f"Resize failed: {str(e)}")
                continue
        
        # Extract coordinates
        y_coords, x_coords = np.where(mask_bool)
        
        if len(y_coords) == 0:
            print(f"No pixels found for mask {i}")
            continue
            
        # Make sure coordinates are within image bounds
        valid = (y_coords < image_np.shape[0]) & (x_coords < image_np.shape[1])
        y_coords, x_coords = y_coords[valid], x_coords[valid]
        
        if len(y_coords) == 0:
            print(f"No valid pixels found for mask {i}")
            continue
            
        try:
            # Extract pixel values
            pixel_values = image_np[y_coords, x_coords]
            print(f"Extracted {len(pixel_values)} pixels for mask {i}")
            
            # Save individual lane data
            lane_file = os.path.join(output_dir, f'lane_{i}_{labels[i]}.npy')
            
            # Save data in a structured format
            lane_data = {
                'label': labels[i],
                'coordinates': np.array(list(zip(y_coords, x_coords))),
                'pixel_values': pixel_values,
                'mean_value': np.mean(pixel_values, axis=0) if len(pixel_values) > 0 else None,
                'std_value': np.std(pixel_values, axis=0) if len(pixel_values) > 0 else None
            }
            
            np.save(lane_file, lane_data)
            print(f"Saved data for lane {i} to {lane_file}")
            
        except Exception as e:
            print(f"Error processing lane {i}: {str(e)}")
    
    return lane_data

def get_mask_pixel_coordinates(masks):
    # List to store pixel coordinates for each mask
    mask_coordinates = []
    
    for mask in masks:
        # Find the indices where mask == 1
        # This returns a tuple of two arrays: 
        # - first array contains y-coordinates (row indices)
        # - second array contains x-coordinates (column indices)
        y_coords, x_coords = np.where(mask == 1)
        
        # Combine coordinates into a list of (x, y) tuples
        coordinates = list(zip(x_coords, y_coords))
        
        mask_coordinates.append(coordinates)
    
    return mask_coordinates

# def draw_segmentation_map(image, masks, boxes, labels, args):
#     alpha = 1.0
#     beta = 1.0 # transparency for the segmentation map
#     gamma = 0.0 # scalar added to each sum
#     #convert the original PIL image into NumPy format
#     image = np.array(image)
#     # convert from RGN to OpenCV BGR format
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     for i in range(len(masks)):
#         # apply a randon color mask to each object
#         color = COLORS[coco_names.index(labels[i])]
#         print(COLORS)
#         if masks[i].any() == True:
#             red_map = np.zeros_like(masks[i]).astype(np.uint8)
#             green_map = np.zeros_like(masks[i]).astype(np.uint8)
#             blue_map = np.zeros_like(masks[i]).astype(np.uint8)
#             red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
#             # combine all the masks into a single image
#             segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
#             # apply mask on the image
#             cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

#             lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
#             tf = max(lw - 1, 1) # Font thickness.
#             p1, p2 = boxes[i][0], boxes[i][1]
#             if not args.no_boxes:
#                 # draw the bounding boxes around the objects
#                 cv2.rectangle(
#                     image, 
#                     p1, p2, 
#                     color=color, 
#                     thickness=lw,
#                     lineType=cv2.LINE_AA
#                 )
#                 w, h = cv2.getTextSize(
#                     labels[i], 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     fontScale=lw / 3, 
#                     thickness=tf
#                 )[0]  # text width, height
#                 w = int(w - (0.20 * w))
#                 outside = p1[1] - h >= 3
#                 p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#                 # put the label text above the objects
#                 cv2.rectangle(
#                     image, 
#                     p1, 
#                     p2, 
#                     color=color, 
#                     thickness=-1, 
#                     lineType=cv2.LINE_AA
#                 )
#                 cv2.putText(
#                     image, 
#                     labels[i], 
#                     (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     fontScale=lw / 3.8, 
#                     color=(255, 255, 255), 
#                     thickness=tf, 
#                     lineType=cv2.LINE_AA
#                 )
#     return image
def draw_segmentation_map(image, masks, boxes, labels, args):
    alpha = 1.0
    beta = 1.0 # transparency for the segmentation map
    gamma = 0.0 # scalar added to each sum
    #convert the original PIL image into NumPy format
    image = np.array(image)
    # convert from RGN to OpenCV BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask_pixel_coords = get_mask_pixel_coordinates(masks)
    # for i, coords in enumerate(mask_pixel_coords):
    #     with open(f'mask_{i}_coordinates.txt', 'w') as f:
    #         for coord in coords:
    #             f.write(f"{coord[0]},{coord[1]}\n")

                
    for i in range(len(masks)):
        # apply a randon color mask to each object
        color = COLORS[coco_names.index(labels[i])]
        print(color)
        if masks[i].any() == True:
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

            lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
            tf = max(lw - 1, 1) # Font thickness.
            p1, p2 = boxes[i][0], boxes[i][1]
            if not args.no_boxes:
                # draw the bounding boxes around the objects
                cv2.rectangle(
                    image, 
                    p1, p2, 
                    color=color, 
                    thickness=lw,
                    lineType=cv2.LINE_AA
                )
                w, h = cv2.getTextSize(
                    labels[i], 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=lw / 3, 
                    thickness=tf
                )[0]  # text width, height
                w = int(w - (0.20 * w))
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # put the label text above the objects
                cv2.rectangle(
                    image, 
                    p1, 
                    p2, 
                    color=color, 
                    thickness=-1, 
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    image, 
                    labels[i], 
                    (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=lw / 3.8, 
                    color=(255, 255, 255), 
                    thickness=tf, 
                    lineType=cv2.LINE_AA
                )
    return image, mask_pixel_coords