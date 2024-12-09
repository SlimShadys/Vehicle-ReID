from PIL import Image
from tqdm import tqdm
from misc.utils import is_incomplete
from reid.models.color_model import EfficientNet

# Description: This script filters out the bounding boxes that are stationary, have the wrong color or have incomplete bounding boxes.
def filter_frames(full_dict, reid_model, car_classifier, transforms,
                  device, perf_timer, target_colors, stationary_threshold,
                  min_stationary_frames, min_box_size, frame_width, frame_height):

    # Variables to track removals
    frames_removed = {
        "color": 0,
        "stationary": 0,
        "incomplete": 0
    }
    faulty_ids = []

    for id, frames in tqdm(full_dict.items(), desc="Filtering Bounding Boxes", total=len(full_dict)):

        # Skip if there are no frames
        if len(frames) == 0:
            faulty_ids.append(id)
            continue

        # Check for stationary vehicles
        # We check if there are at least 2 frames to calculate the movement, otherwise we skip directly to the Incomplete Filtering
        if len(frames) > 1:
            is_stationary = stationary_filtering(frames, (frame_width, frame_height), stationary_threshold, min_stationary_frames)
            
            # If the vehicle is stationary, remove all the frames and move to the next ID
            if is_stationary:
                frames_removed["stationary"] += len(frames)
                faulty_ids.append(id)
                continue

        # Filter incomplete bounding boxes for same cars
        removed_incomplete, filtered_same_cars = incomplete_filtering(min_box_size, frame_width, frame_height, frames)
        frames_removed["incomplete"] += removed_incomplete

        # Check if there are valid frames left for both situations
        # If there are no valid frames, mark the vehicle as faulty and skip to the next ID
        if len(filtered_same_cars) == 0:
            faulty_ids.append(id)   # Mark vehicle as faulty
            continue                # Skip to the next ID
        else: # Perform color filtering for same cars only if there are valid frames
            removed_color, filtered_same_cars = color_filtering(id, filtered_same_cars, reid_model, car_classifier, transforms, device, target_colors)
            frames_removed["color"] += removed_color

            # If there are still valid frames after color filtering, update same cars dict
            if len(filtered_same_cars) > 0:
                full_dict[id] = filtered_same_cars
            else:
                faulty_ids.append(id)
        perf_timer.register_call("Main filtering")

    # Remove vehicles with stationary, incomplete, or no valid frames
    print("Removing faulty vehicles (Stationary, wrong color, or incomplete bounding boxes):")
    for id in faulty_ids:
        print(f"\t- Removing vehicle with ID {id}")
        del full_dict[id]

    return full_dict, frames_removed

def color_filtering(id, frames, reid_model, car_classifier, transforms, device, target_colors):
    # Exit immediately if there are no target colors
    if target_colors is None or len(target_colors) == 0 or target_colors == ['']:
        return 0, frames

    frames_removed_color = 0
    timestamps_to_remove = []

    # Extract cropped images from the full_dict for the current ID
    cropped_images = [frame['cropped_image'] for frame in frames]
    
    # Run inference on the Color Model one frame at a time
    for i in tqdm(range(len(cropped_images)), desc=f"Color filtering for ID: {id}", total=len(cropped_images)):
        orig_frame = cropped_images[i]
        frame = Image.fromarray(orig_frame)                 # Convert to PIL Image
        frame = transforms.val_transform(frame)             # Convert to torch.Tensor
        frame = frame.unsqueeze(0).float().to(device)       # Convert to Tensor and send to device

        # EfficientNet
        if isinstance(car_classifier, EfficientNet):
            prediction = car_classifier.predict(frame)
            color_prediction = [entry['color'] for entry in prediction]
        # ResNet + SVM
        else:
            svm_embedding = reid_model(frame).detach().cpu().numpy()
            color_prediction = car_classifier.predict(svm_embedding)

        if not any(c in target_colors for c in color_prediction):
            # Collect the timestamp of the frame to remove
            timestamps_to_remove.append(frames[i]['timestamp'])
            frames_removed_color += 1 # Increment the number of frames removed

    # Remove frames with matching timestamps outside the loop for efficiency
    frames = [frame for frame in frames if frame['timestamp'] not in timestamps_to_remove]

    return frames_removed_color, frames

def incomplete_filtering(min_box_size, frame_width, frame_height, frames):
    # Counter
    frames_removed_incomplete = 0

    # Filter for incomplete bounding boxes
    initial_frame_count = len(frames)
    filtered_frames = [frame for frame in frames if not is_incomplete(frame['bounding_box'], min_box_size, frame_width, frame_height, margin=50, center_threshold=0.2)]
    filtered_frame_count = len(filtered_frames)  # Count frames after filtering

    # Calculate how many frames were removed
    frames_removed_incomplete += initial_frame_count - filtered_frame_count
        
    return frames_removed_incomplete, filtered_frames

def stationary_filtering(frames, frame_size, stationary_threshold, min_stationary_frames):
    # Variables
    frame_width, frame_height = frame_size
    stationary_frame_count = 0
    centroids = []

    # Calculate the centroids of the bounding boxes over time
    for frame in frames:
        tx, ty, w, h = frame['bounding_box'] # Get bounding box coordinates (tlwh)
        # Calculate centroid
        centroid_x = tx + w / 2
        centroid_y = ty + h / 2
        centroids.append((centroid_x, centroid_y))

    # Calculate the movement of the centroid between frames
    for i in range(1, len(centroids)):
        dist_x = abs(centroids[i][0] - centroids[i-1][0])
        dist_y = abs(centroids[i][1] - centroids[i-1][1])
        
        # If the movement is below a threshold, count the frame as stationary
        if dist_x <= stationary_threshold * frame_width and dist_y <= stationary_threshold * frame_height:
            stationary_frame_count += 1
    
    # If the vehicle is stationary for too many frames, remove it
    if stationary_frame_count > min_stationary_frames:
        return True
    else:
        return False