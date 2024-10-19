import os
import random
import shutil
import zlib

import bson
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from PIL import Image
from tqdm import tqdm

# Retrieve and decompress the data from MongoDB
def decompress_img(compressed_data, shape, dtype):
    decompressed_data = zlib.decompress(compressed_data)
    array = np.frombuffer(decompressed_data, dtype=dtype)  # Convert bytes back to NumPy array
    return array.reshape(shape)  # Reshape to original dimensions

# Compress the image data before storing in MongoDB
def compress_img(img):
    compressed_data = zlib.compress(img.tobytes())
    compressed_data = bson.Binary(compressed_data)
    return compressed_data

def is_incomplete(box, min_box_size, frame_width, frame_height, margin=50, center_threshold=0.2):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Center of the bounding box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    
    # Define the central region of the frame
    center_x_min = frame_width * (center_threshold / 2)
    center_x_max = frame_width * (1 - center_threshold / 2)
    center_y_min = frame_height * (center_threshold / 2)
    center_y_max = frame_height * (1 - center_threshold / 2)
    
    # Check if the bounding box is smaller than the minimum size
    if (box_width < min_box_size or box_height < min_box_size):
        return True
    
    # Check if the bounding box extends beyond the frame boundaries
    if (x1 < margin or y1 < margin or x2 > (frame_width - margin) or y2 > (frame_height - margin)):
        return True
    
    # # Check if the bounding box is near the center
    if not (box_center_x > center_x_min and box_center_x < center_x_max and box_center_y > center_y_min and box_center_y < center_y_max):
        return True
    
    return False

# Function to load the middle frame of a vehicle's bounding boxes
def load_middle_frame(bbox_output_path, vehicle_id, camera_id):
    vehicle_id = str(vehicle_id).split('_V')[1]  # Extract the vehicle ID from the unique ID
    vehicle_dir = os.path.join(bbox_output_path, f'Camera-{camera_id}', str(vehicle_id))
    frame_files = sorted(os.listdir(vehicle_dir))
    if vehicle_id == '4':
        frame_path = os.path.join(vehicle_dir, frame_files[0])
    else:
        middle_frame_index = len(frame_files) // 2  # Pick the middle frame
        frame_path = os.path.join(vehicle_dir, frame_files[middle_frame_index])
    img = Image.open(frame_path).convert("RGB")
    img = img.resize((128, 128))
    return img

def create_mask(image, image_path):
    # Extract name of the image file
    image_name = os.path.basename(image_path).split('.')[0]

    # Extract directory of the image file
    image_dir = os.path.dirname(image_path)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create an empty mask with the same height and width as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # List to store points of the polygon
    roi_points = []

    # Mouse callback function to capture points
    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a small green circle where you click
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            cv2.imshow("Select ROI", image)

    # Show the image and set the mouse callback
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select ROI", image)
    cv2.setMouseCallback("Select ROI", select_points)

    print("Select the ROI by clicking points, and press 'Enter' when done. Press 'Esc' to cancel.")

    # Wait for the user to press a key
    key = cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()

    # If 'Esc' is pressed (key == 27), return None or raise an exception if no points were drawn
    if key == 27:
        if len(roi_points) < 2:
            raise Exception("Esc pressed or not enough points selected. Operation canceled")

    # If there are enough points, create a polygon mask
    if len(roi_points) > 2:
        roi_polygon = np.array([roi_points], dtype=np.int32)
        cv2.fillPoly(mask, roi_polygon, 255)  # Fill the polygon on the mask with white

        # Save the mask if needed
        mask_path = os.path.join(image_dir, 'roi_masks', f"{image_name}_roi_mask.png")
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved at: {mask_path}")

        # You can also display the mask
        cv2.namedWindow("ROI Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("ROI Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return mask
    else:
        raise Exception("Not enough points were selected to form a polygon.")

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_prime, y1_prime, x2_prime, y2_prime = box2

    xi1 = max(x1, x1_prime)
    yi1 = max(y1, y1_prime)
    xi2 = min(x2, x2_prime)
    yi2 = min(y2, y2_prime)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_prime - x1_prime) * (y2_prime - y1_prime)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
       
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y, use_amp=False, train=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    if use_amp and train is True:
        dist = dist.half()
    else:
        dist = dist.type(torch.float32) # Force the type to float32 if not using AMP
        x = x.type(torch.float32)
        y = y.type(torch.float32)
            
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt() # for numerical stability
    return dist

def save_model(state, output_dir, model_path='model.pth'):
    '''
        Save the model to a file
        @param state: The state of the model, optimizer, scheduler, and loss value
        @param output_dir: The output directory to save the model
        @param model_path: The file name of the model to save
    '''
    torch.save(state, os.path.join(output_dir, model_path))
    
def load_model(model_path: str,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler_warmup: torch.optim.lr_scheduler._LRScheduler,
               device):
    '''
    Load the model from a file
    @param model_path: The path to the model file
    @param model: The model to load
    @param optimizer: The optimizer to load
    @param scheduler_warmup: The scheduler to load
    @param device: The device to load the model to
    '''
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_warmup.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Correctly loaded the model, optimizer, and scheduler from: {model_path}")
    return model, optimizer, scheduler_warmup, start_epoch

# Get the number of pids, images, and cameras
# Needed for printing the dataset statistics
def get_imagedata_info(data):
    car_ids, cam_ids = [], []
    # img_path, folder, car_id, cam_id, model_id, color_id, type_id, timestamp
    for _, _, car_id, cam_id, _, _, _, _ in data:
        car_ids += [car_id]
        cam_ids += [cam_id]
    car_ids = set(car_ids)
    cam_ids = set(cam_ids)
    num_car_ids = len(car_ids)
    num_cam_ids = len(cam_ids)
    num_imgs = len(data)
    return num_car_ids, num_imgs, num_cam_ids

# Search function for getting the index of the data
# Needed for RPTM Training
def search(path):
    start = 0
    count = 0
    data_index = []
    for i in sorted(os.listdir(path)):
        x = len(os.listdir(os.path.join(path, i)))
        data_index.append((count, start, start + x-1))
        count = count + 1
        start = start + x
    return data_index

# Needed for RPTM Training
def strint(x, dataset):
    if dataset == 'veri_776':
        if len(str(x))==1:
            return '00'+str(x)
        if len(str(x))==2:
            return '0'+str(x)
        if len(str(x))==3:
            return str(x)
    elif dataset == 'vehicle_id' or dataset == 'vru' or dataset == 'veri_wild':
        if len(str(x))==1:
            return '0000'+str(x)
        if len(str(x))==2:
            return '000'+str(x)
        if len(str(x))==3:
            return '00'+str(x)
        if len(str(x))==4:
            return '0'+str(x)
        if len(str(x))==5:
            return str(x)
    else:
        raise Exception('Invalid dataset name')

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, dataset_name='veri_776', re_rank=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
        
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    
    description = "Computing CMC and mAP" if not re_rank else "Computing CMC and mAP with re-ranking"
    for q_idx in tqdm(range(num_q), desc=description, total=num_q):
        if dataset_name == 'vehicle_id' or dataset_name == 'vru':
            remove = False  # without camid imformation remove no images in gallery
        else:
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, all_AP

def visualize_ranked_results(distmat, dataset, save_dir='log/ranked_results', topk=10):
    """
    Visualize ranked results
    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape
    query, gallery = dataset
    indices = np.argsort(distmat, axis=1)

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = os.path.join(dst, prefix + '_top' + str(rank).zfill(3))

            if not os.path.exists(dst): os.makedirs(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = os.path.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + os.path.basename(src))
            shutil.copy(src, dst)

    for q_idx in tqdm(range(num_q), desc="Computing visualized ranked results", total=num_q):
        # Simply copy the query image to the output directory
        qimg_path, _, qpid, qcamid, _, _, _, _ = query[q_idx]
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = os.path.join(save_dir, os.path.basename(qimg_path[0]))
        else:
            qdir = os.path.join(save_dir, os.path.basename(qimg_path))
            
        if not os.path.exists(qdir): os.makedirs(qdir)
        
        _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        # Copy the ranked images to the output directory
        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, _, gpid, gcamid, _, _, _, _ = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                if rank_idx > topk:
                    break
    print("Done")

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    """
    Created on Fri, 25 May 2018 20:29:09

    @author: luohao
    """

    """
    CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
    url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
    Matlab version: https://github.com/zhunzhong07/person-re-ranking
    """

    """
    API

    probFea: all feature vectors of the query set (torch tensor)
    probFea: all feature vectors of the gallery set (torch tensor)
    k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
    MemorySave: set to 'True' when using MemorySave mode
    Minibatch: avaliable when 'MemorySave' is 'True'
    """
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
                  torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        
        # Force the type to float32
        distmat = distmat.type(torch.float32)
        feat = feat.type(torch.float32)

        distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
        
        original_dist = distmat.cpu().numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    total_steps = all_num + query_num  # Total steps for both re-ranking and Jaccard distance computation
    with tqdm(total=total_steps, desc="Computing re-ranking") as pbar:
        for i in range(all_num):
            # K-Reciprocal Neighbors
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(k1 / 2)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
            pbar.update(1)

        original_dist = original_dist[:query_num, ]
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float16)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(gallery_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
            pbar.update(1)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def load_labels(label_file):
    label = []
    with open(label_file, "r", encoding='cp1251') as ins:
        for line in ins:
            label.append(line.rstrip())

    return label

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Correctly set the seed to:", seed)