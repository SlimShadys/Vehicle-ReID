import numpy as np
import torch
from tqdm import tqdm

from db.database import Database
from misc.utils import cosine_similarity_np, normalize_np

def get_max_similarity_per_vehicle(reid_DB: Database, reid_cfg, camera, device, similarity_algorithm, similarity_method):
    # After the pipeline is done, we must refine the BBoxes as to:
    # - Unify the BBoxes of the same vehicle that have been categorized as different vehicles
    #   -> We can think of using the ReID model to check if the BBoxes are from the same vehicle
    camera_id = camera['_id']
    camera_vehicles = reid_DB.get_camera_frames(camera_id=camera_id)

    if len(camera_vehicles) == 0:
        print(f"WARNING! No vehicles found for camera {camera_id}. Skipping unification...")
        return []

    # Initialize dictionary to store the max similarity for each vehicle
    max_similarity_per_vehicle = []

    if similarity_method == 'mean':
        # Construct a similarity matrix for the vehicles from the same camera
        # We will compare the features of the vehicles from the same camera
        # If the features are similar, we will consider them as the same vehicle
        # If the features are different, we will consider them as different vehicles
        mean_embeddings_dict_np = {}

        # Extract embeddings and compute mean per vehicle
        for vehicle, frames in camera_vehicles.items():
            # NumPy version
            mean_embeddings_dict_np[(camera_id, vehicle)] = np.mean([np.array(frame[0]) for _, frame in frames.items()], axis=0)
            
        # Convert embeddings to tensor and normalize if needed
        vehicle_ids = list(mean_embeddings_dict_np.keys())
        mean_embeddings = np.stack([mean_embeddings_dict_np[vid] for vid in vehicle_ids])
        
        # Normalize the embeddings if necessary
        if reid_cfg.TEST.NORMALIZE_EMBEDDINGS:
            mean_embeddings = np.array([normalize_np(embedding) for embedding in mean_embeddings])

        # Compute pairwise similarity for vehicles
        f = torch.Tensor(mean_embeddings).to(device)
        sim = torch.matmul(f, f.T).cpu().numpy()
        n = len(vehicle_ids)

        # init merge queue
        for i in range(n):
            max_similarity_matmul = -float('inf')
            best_j_idx = None
            i_idx = int(vehicle_ids[i][1].split("_V")[1])

            for j in range(n):
                j_idx = int(vehicle_ids[j][1].split("_V")[1])

                # Skip if i_idx and j_idx refer to the same vehicle ID
                if i_idx == j_idx:
                    continue

                # Update max similarity and best pair if the condition is met
                if sim[i, j] > max_similarity_matmul:
                    max_similarity_matmul = sim[i, j]
                    best_j_idx = j_idx

            # Append to merge queue only if a valid best_j_idx is found
            if best_j_idx is not None:
                # Vehicle_ID, Vehicle_ID_MaxSim, Similarity
                max_similarity_per_vehicle.append((i_idx, best_j_idx, max_similarity_matmul))

    elif similarity_method == 'individual':
        # Vehicle IDs list
        vehicle_ids = list(camera_vehicles.keys())

        # Number of vehicles
        num_vehicles = len(vehicle_ids)

        # Normalize the embeddings from camera_vehicles
        for vehicle in camera_vehicles:
            for frame in camera_vehicles[vehicle]:
                camera_vehicles[vehicle][frame][0] = normalize_np(np.array(camera_vehicles[vehicle][frame][0])).tolist()

        # Compare all vehicle pairs
        for i in tqdm(range(num_vehicles), desc="Computing Similarity Matrix"):
            vehicle_i = vehicle_ids[i]
            max_similarity = -float('inf')  # Initialize with very low similarity
            max_vehicle_pair = None         # Store the pair with the highest similarity

            for j in range(i + 1, num_vehicles):  # Only compute upper triangle
                vehicle_j = vehicle_ids[j]

                similarities = []  # Store individual frame-to-frame similarities

                # Compare each frame of vehicle_i with each frame of vehicle_j
                for frame_i in camera_vehicles[vehicle_i]:
                    for frame_j in camera_vehicles[vehicle_j]:
                        emb_i = np.array(camera_vehicles[vehicle_i][frame_i][0])  # [2048]
                        emb_j = np.array(camera_vehicles[vehicle_j][frame_j][0])  # [2048]

                        # Compute similarity using NumPy
                        if similarity_algorithm == 'cosine':
                            similarity = cosine_similarity_np(emb_i, emb_j)
                        elif similarity_algorithm == 'euclidean':
                            similarity = np.linalg.norm(emb_i - emb_j)

                        similarities.append(similarity)

                # Compute the average similarity between the two vehicles
                avg_similarity = np.mean(similarities)

                # Check if this is the highest similarity encountered for this vehicle
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    max_vehicle_pair = vehicle_j

            # Store the vehicle pair with the highest similarity
            if max_vehicle_pair:
                # Vehicle_ID, Vehicle_ID_MaxSim, Similarity
                max_similarity_per_vehicle.append((vehicle_i, max_vehicle_pair, max_similarity))

    elif similarity_method == 'area_avg':
        mean_embeddings_dict_np = {}

        # Extract embeddings and compute mean per vehicle
        for vehicle, frames in camera_vehicles.items():
            if (camera_id, vehicle) not in mean_embeddings_dict_np:
                mean_embeddings_dict_np[(camera_id, vehicle)] = []

            # Extract bounding box areas
            bboxes = [frame[4] for _, frame in frames.items()]
            div = min(map(lambda x: x[2] * x[3], bboxes))  # Reference area

            weighted_embeddings = np.zeros_like(np.array(list(frames.values())[0][0]))  # Store weighted embeddings

            for _, frame in frames.items():
                embedding = frame[0]  # Shape [2048]
                area = frame[4][2] * frame[4][3]  # Width * Height of bbox
                weighted_embeddings += np.array(embedding) * (area / div)  # Apply weight

            # Normalize the weighted embeddings
            weighted_embeddings /= np.linalg.norm(weighted_embeddings)

            # Compute the mean embedding (average along axis=0)
            mean_embeddings_dict_np[(camera_id, vehicle)] = weighted_embeddings  # Shape [2048]

        # Convert embeddings to tensor and normalize if needed
        vehicle_ids = list(mean_embeddings_dict_np.keys())
        mean_embeddings = np.stack([mean_embeddings_dict_np[vid] for vid in vehicle_ids])

        # Compute pairwise similarity for vehicles
        f = torch.Tensor(mean_embeddings).to(device)
        sim = torch.matmul(f, f.T).cpu().numpy()
        n = len(vehicle_ids)

        # init merge queue
        for i in range(n):
            max_similarity_matmul = -float('inf')
            best_j_idx = None
            i_idx = int(vehicle_ids[i][1].split("_V")[1])

            for j in range(n):
                j_idx = int(vehicle_ids[j][1].split("_V")[1])

                # Skip if i_idx and j_idx refer to the same vehicle ID
                if i_idx == j_idx:
                    continue

                # Update max similarity and best pair if the condition is met
                if sim[i, j] > max_similarity_matmul:
                    max_similarity_matmul = sim[i, j]
                    best_j_idx = j_idx

            # Append to merge queue only if a valid best_j_idx is found
            if best_j_idx is not None:
                # Vehicle_ID, Vehicle_ID_MaxSim, Similarity
                max_similarity_per_vehicle.append((i_idx, best_j_idx, (max_similarity_matmul)))
    else:
        raise ValueError(f"Invalid similarity method: {similarity_method}")
    
    return max_similarity_per_vehicle