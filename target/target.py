import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.linalg import norm
from PIL import Image, ImageOps
from torchvision import transforms as T
from tqdm import tqdm
from yacs.config import CfgNode

from misc.utils import decompress_img
from reid.datasets.transforms import Transformations
from reid.model import ModelBuilder
from tracking.cameras import CameraLayout

num_classes = {
    'ai_city': 440,
    'ai_city_sim': 1802,
    'vehicle_id': 13164,
    'veri_776': 576,
    'veri_wild': 30671,
    'vric': 2811,
    'vru': 7086,
}

def multitrack_search(features, final_tracks, target_cam, K=5):

    # Convert the target feature to a numpy array
    features = features.detach().cpu().numpy() # Convert to numpy array

    # initialize a similarity matrix based on how many entries are there in the final_tracks
    similarity_matrix = np.zeros(len(final_tracks))

    # iterate over the final_tracks
    # in a single trajectory, we must remove the Tracks which belong to the same camera as the Target image
    track_removed = False
    to_remove_trajectories = []

    for i, trajectory in tqdm(enumerate(final_tracks), desc="Searching for target in tracks", total=len(final_tracks)):
        # Filter out tracks from the same camera as the target image
        initial_track_count = len(trajectory.tracks)

        # There could be a case where the target_cam is None, so we don't need to filter out any tracks
        if target_cam is not None:
            trajectory.tracks = [track for track in trajectory.tracks if track.cam != target_cam]

        # Check if any tracks were removed
        if len(trajectory.tracks) < initial_track_count:
            track_removed = True

        # Remove the trajectory if it has no tracks
        if len(trajectory.tracks) == 0:
            to_remove_trajectories.append((trajectory, i))
            continue

        # Recompute the mean feature of the trajectory only if a track was removed
        if track_removed:
            mean_feats = [track._mean_feature for track in trajectory.tracks]
            trajectory._mean_feature = np.mean(mean_feats, axis=0)
            # trajectory.mean_feature /= np.linalg.norm(trajectory.mean_feature)  # Normalize
            track_removed = False

        # Get the mean features of the trajectory
        trajectory_features = trajectory.mean_feature

        # calculate the similarity between the target and the track
        similarity = np.dot(features, trajectory_features) / ((norm(features) * norm(trajectory_features)))
        similarity_matrix[i] = similarity

    # Remove trajectories and update similarity matrix
    for trajectory, idx in to_remove_trajectories:
        final_tracks.remove(trajectory)
        similarity_matrix[idx] = -1

    # Retrieve the top-K most similar tracks
    top_k_indices = np.argsort(similarity_matrix)[::-1][:K]

    # Get the top-K most similar tracks
    top_k_tracks = [final_tracks[i] for i in top_k_indices]

    # Filter the similarity matrix based on the top-K indices
    similarity_matrix = similarity_matrix[top_k_indices]

    return similarity_matrix, top_k_tracks

def singletrack_search(target_feat: torch.Tensor, all_frames_tracks, target_cam, K, device='cpu'):
    if target_cam is not None:
        # filter out the tracks from the same camera as the target image
        all_frames_tracks = [track for track in all_frames_tracks if track.cam != target_cam]

    # Get the features of all the tracks
    f = torch.Tensor(np.stack([track._mean_feature for track in all_frames_tracks])).to(device)

    # Compute similarity matrix
    sim_matrix = torch.matmul(target_feat, f.T).squeeze()

    # Retrieve the top-K most similar tracks (Torch version)
    top_k_indices = torch.argsort(sim_matrix, descending=True).squeeze()[:K]

    # Get the top-K most similar tracks
    top_k_tracks = [all_frames_tracks[i] for i in top_k_indices]

    # Filter the similarity matrix based on the top-K indices
    sim_matrix = sim_matrix[top_k_indices]

    return sim_matrix, top_k_tracks

def cameralink_search(
    target_feat: torch.Tensor, all_frames_tracks, cams, K, device='cpu'
):
    """
    Searches for tracks in multiple cameras and creates a trajectory by linking features across cameras.

    Args:
        target_feat (torch.Tensor): The feature of the target track.
        all_frames_tracks (list): List of tracks containing features and metadata.
        cams (object): Object containing the number of cameras (`n_cams`).
        target_cam (int): The ID of the target's originating camera.
        K (int): The number of top matches to consider per camera.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        trajectory (dict): A dictionary where keys are indices (0, 1, 2) representing 
                           the top-K tracks, and values are lists of tracks across cameras.
    """

    # Initialize the camera-specific tracks
    camera_tracks = [[] for _ in range(cams.n_cams)]
    for track in all_frames_tracks:
        camera_tracks[track.get_cam_id()].append(track)

    # ==== FIRST CAMERA ONLY ====
    # Initialize the trajectory dictionary
    trajectory = {i: [] for i in range(K)}

    # Get the features of all the tracks from Camera 0
    f = torch.Tensor(np.stack([track._mean_feature for track in camera_tracks[0]])).to(device)

    # Compute similarity matrix
    sim_matrix = torch.matmul(target_feat, f.T).squeeze()

    # Retrieve the top-K most similar tracks
    top_k_indices = torch.argsort(sim_matrix, descending=True).squeeze()

    # Ensure top_k_indices is 1D
    if top_k_indices.dim() == 0:  # If it's a 0D tensor
        top_k_indices = top_k_indices.unsqueeze(0)  # Make it 1D

    # Handle the case where the number of indices is less than K
    top_k_indices = top_k_indices[:K] if len(top_k_indices) >= K else top_k_indices

    # Get the top-K most similar tracks
    top_k_tracks = [camera_tracks[0][i] for i in top_k_indices.tolist()]
    
    # Convert Target Feature to a numpy array
    target_feat = target_feat.detach().cpu().numpy()

    # Append the top-K tracks to the trajectory
    for i, track in enumerate(top_k_tracks):
        trajectory[i].append((track, sim_matrix[i].item()))

    # ==== OTHER CAMERAS ====
    for cam_id in range(1, cams.n_cams):

        # Skip if there are no tracks in the current camera
        if len(camera_tracks[cam_id]) == 0:
            continue

        # Get the features of all the tracks for this camera
        f = torch.Tensor(np.stack([track._mean_feature for track in camera_tracks[cam_id]])).to(device)

        for i in range(K):

            # Check if there are enough trajectories to process
            if i >= len(trajectory):
                break

            # Ensure the trajectory[i] has elements
            if not trajectory[i]:
                continue

            k_feat_mean = torch.Tensor(np.mean([track._mean_feature for track, _ in trajectory[i]], axis=0)).to(device)

            # Compute similarity matrix
            sim_matrix = torch.matmul(k_feat_mean, f.T).squeeze()

            # Handle case where sim_matrix could be 0D
            if sim_matrix.dim() == 0:
                sim_matrix = sim_matrix.unsqueeze(0)

            # Retrieve the top most similar track
            top_track_idx = torch.argmax(sim_matrix)

            if top_track_idx >= len(camera_tracks[cam_id]):
                continue  # Skip if the index is invalid

            top_track = camera_tracks[cam_id][top_track_idx]
            
            # Append the top track to the trajectory along with the similarity score
            trajectory[i].append((top_track, sim_matrix[top_track_idx].item()))

    return trajectory

def target_search(target: dict, final_trajectories: list, all_frames_tracks: list, cams: CameraLayout, cfg: CfgNode):
    # Extract configs
    augmentation_cfg = cfg.REID.AUGMENTATION
    device = cfg.MISC.DEVICE
    gpu_id = cfg.MISC.GPU_ID
    dataset_name = cfg.REID.DATASET.DATASET_NAME
    target_search_method = cfg.MTMC.TARGET_SEARCH_METHOD
    K = 5

    # CUDA device
    if device == 'cuda' and torch.cuda.is_available():
        device = f"{device}:{gpu_id}"
    else:
        device = 'cpu'

    # Get the Re-ID model
    reid_model = ModelBuilder(
        model_name=cfg.REID.MODEL.NAME,
        pretrained=cfg.REID.MODEL.PRETRAINED,
        num_classes=num_classes[dataset_name],
        model_configs=cfg.REID.MODEL,
        device=device
    ).move_to(device)
    reid_model.eval()

    # Load parameters from a .pth file
    val_path = cfg.REID.TEST.MODEL_VAL_PATH
    if (val_path is not None):
        print("Loading model parameters from file...")
        if ('ibn' in cfg.REID.MODEL.NAME):
            reid_model.load_param(val_path)
        else:
            checkpoint = torch.load(val_path, map_location=device)
            reid_model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded model parameters from file: {val_path}")

    # Initialize the transformations
    transforms = Transformations(dataset=None, configs=augmentation_cfg)

    # ============= TARGET SEARCH =============
    target_path, target_cam = target["path"], target["camera"]
    target_image = Image.open(target_path).convert("RGB")

    # Get the features of the target image
    frame = transforms.val_transform(target_image)             # Convert to torch.Tensor
    frame = frame.unsqueeze(0).float().to(device)

    target_feat = reid_model.forward(frame, training=False)
    target_feat = torch.nn.functional.normalize(target_feat, dim=1, p=2)

    if target_search_method == 'multitrack_search':
        similarities, top_k_tracks = multitrack_search(target_feat, final_trajectories, target_cam, K) # Run the Multi-Track Search method
    elif target_search_method == 'singletrack_search':
        similarities, top_k_tracks = singletrack_search(target_feat, all_frames_tracks, target_cam, K, device) # Run the Single-Track Search method
    elif target_search_method == 'cameralink_search':
        top_k_tracks = cameralink_search(target_feat, all_frames_tracks, cams, K, device) # Run the CameraLink Search method
    else:
        raise ValueError(f"Invalid target search method: {target_search_method}")

    # Prepare image transforms
    query_trans = T.Pad(4, 0)
    good_trans = T.Pad(4, (0, 255, 0))          # Green
    bad_trans = T.Pad(4, (255, 0, 0))           # Red
    empty_trans = T.Pad(4, (255, 255, 0, 255))  # Yellow
    min_bbox_area = 100

    # Prepare target image
    q_image = ImageOps.contain(target_image, (128, 128))

    # Create transparent image for missing views
    width, height = 128, 128
    transparent_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    transparent_img = empty_trans(transparent_img)

    # Print out which are the top-K most similar tracks
    print(f"Top-{K} most similar tracks:")
    if target_search_method == 'multitrack_search':

        # Create a dictionary to store images for each trajectory and camera
        # Structure: traj_images[trajectory_idx][camera_id] = image
        traj_images = {}

        # Process top-K tracks
        for i, trajectory in enumerate(top_k_tracks):
            
            traj_images[i] = {}  # Initialize dictionary for this trajectory

            tracks = trajectory.tracks
            
            print(f"Trajectory {i+1} | Traj Length: {len(tracks)} | Similarity: {similarities[i]:.4f}")
            
            for track in tracks:
                print(f"\t- VID: {track.vehicle_id} | Camera: {track.cam} | Start Time: {track.start_time:.2f} | End Time: {track.end_time:.2f}")

                # Find a suitable frame to display
                found = False
                for n, (k, v) in enumerate(track.frames.items()):
                    try:  # Handle both DB and Pickle formats
                        shape = v[4] if len(v) > 4 else v['shape']
                    except:
                        continue
                    
                    if shape[0] >= min_bbox_area or shape[1] >= min_bbox_area:
                        idx = k
                        found = True
                        break
                
                if not found:
                    idx = list(track.frames.keys())[len(track.frames) // 2]

                # Extract and process gallery image
                g_frame = track.frames[idx]

                try:  # Pickle format
                    comp_img = g_frame['compressed_image']
                    shape = g_frame['shape']
                except:  # DB format
                    comp_img = g_frame[3]
                    shape = g_frame[4]
                
                g_image = decompress_img(comp_img, shape, dtype=np.uint8)
                g_image = Image.fromarray(g_image).convert("RGB")
                g_image = ImageOps.contain(g_image, (128, 128))

                # Store image indexed by camera ID
                traj_images[i][track.get_cam_id()] = g_image

        # Create figure - rows for trajectories, columns for cameras (+1 for query)
        fig, axes = plt.subplots(len(top_k_tracks), cams.n_cams + 1, figsize=(3*(cams.n_cams + 1), 3*K))

    elif target_search_method == 'singletrack_search':
            
        # Create a dictionary to store images for each trajectory and camera
        # Structure: traj_images[trajectory_idx][camera_id] = image
        traj_images = {}

        # Process top-K tracks
        for i, track in enumerate(top_k_tracks):

            traj_images[i] = {}  # Initialize dictionary for this trajectory

            print(f"VID: {track.vehicle_id} | Camera: {track.cam} | Start Time: {track.start_time:.2f} | End Time: {track.end_time:.2f}")

            # Find a suitable frame to display
            found = False
            for n, (k, v) in enumerate(track.frames.items()):
                try:  # Handle both DB and Pickle formats
                    shape = v[4] if len(v) > 4 else v['shape']
                except:
                    continue
                
                if shape[0] >= min_bbox_area or shape[1] >= min_bbox_area:
                    idx = k
                    found = True
                    break
            
            if not found:
                idx = list(track.frames.keys())[len(track.frames) // 2]

            # Extract and process gallery image
            g_frame = track.frames[idx]

            try:  # Pickle format
                comp_img = g_frame['compressed_image']
                shape = g_frame['shape']
            except:  # DB format
                comp_img = g_frame[3]
                shape = g_frame[4]

            g_image = decompress_img(comp_img, shape, dtype=np.uint8)
            g_image = Image.fromarray(g_image).convert("RGB")
            g_image = ImageOps.contain(g_image, (128, 128))

            # Store image indexed by camera ID
            traj_images[i][track.get_cam_id()] = g_image

        # Create figure - rows for trajectories, columns for cameras (+1 for query)
        fig, axes = plt.subplots(len(top_k_tracks), cams.n_cams + 1, figsize=(3*(cams.n_cams + 1), 3*K))

    else: # cameralink_search

        # Create a dictionary to store images for each trajectory and camera
        # Structure: traj_images[trajectory_idx][camera_id] = image
        traj_images = {}

        # Process top-K tracks
        for i, (k, v) in enumerate(top_k_tracks.items()):
            tracks, mean_sim = [track for track, _ in v], [sim for _, sim in v]
            traj_images[i] = {}  # Initialize dictionary for this trajectory
            
            print(f"Trajectory {i+1} | Traj Length: {len(tracks)} | Mean Similarity: {np.mean(mean_sim):.4f}")
            
            for track in tracks:
                print(f"\t- VID: {track.vehicle_id} | Camera: {track.cam} | Start Time: {track.start_time:.2f} | End Time: {track.end_time:.2f}")

                # Find a suitable frame to display
                found = False
                for n, (k, v) in enumerate(track.frames.items()):
                    try:  # Handle both DB and Pickle formats
                        shape = v[4] if len(v) > 4 else v['shape']
                    except:
                        continue
                    
                    if shape[0] >= min_bbox_area or shape[1] >= min_bbox_area:
                        idx = k
                        found = True
                        break
                
                if not found:
                    idx = list(track.frames.keys())[len(track.frames) // 2]

                # Extract and process gallery image
                g_frame = track.frames[idx]

                try:  # Pickle format
                    comp_img = g_frame['compressed_image']
                    shape = g_frame['shape']
                except:  # DB format
                    comp_img = g_frame[3]
                    shape = g_frame[4]
                
                g_image = decompress_img(comp_img, shape, dtype=np.uint8)
                g_image = Image.fromarray(g_image).convert("RGB")
                g_image = ImageOps.contain(g_image, (128, 128))

                # Store image indexed by camera ID
                traj_images[i][track.get_cam_id()] = g_image

        # Create figure - rows for trajectories, columns for cameras (+1 for query)
        fig, axes = plt.subplots(len(top_k_tracks), cams.n_cams + 1, figsize=(3*(cams.n_cams + 1), 3*K))

    # Set column titles
    axes[0, 0].set_title("Query")
    for j in range(1, cams.n_cams + 1):
        axes[0, j].set_title(f"CAM{j-1}")

    # Turn off all axes
    for ax in axes.flat:
        ax.axis("off")

    # Display query image in first column for each trajectory
    query_img = query_trans(q_image)
    for i in range(len(top_k_tracks)):
        axes[i, 0].imshow(query_img)

    # Display trajectory images
    for traj_idx in range(len(top_k_tracks)):
        for cam_id in range(cams.n_cams):
            if cam_id in traj_images[traj_idx]:
                # Display track image with green border
                img = good_trans(traj_images[traj_idx][cam_id])
                axes[traj_idx, cam_id + 1].imshow(img)
            else:
                # Display transparent image for missing camera views
                axes[traj_idx, cam_id + 1].imshow(transparent_img)

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)