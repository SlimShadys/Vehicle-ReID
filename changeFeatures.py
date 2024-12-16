
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from misc.utils import decompress_img
from pipeline import load_config_and_device, setup_database
from reid.datasets.transforms import Transformations
from reid.model import ModelBuilder

cfg, device = load_config_and_device()

transforms = Transformations(dataset=None, configs=cfg.REID.AUGMENTATION)

model_builder = ModelBuilder(
    model_name=cfg.REID.MODEL.NAME,
    pretrained=cfg.REID.MODEL.PRETRAINED,
    num_classes=576,
    model_configs=cfg.REID.MODEL,
    device=device
)
reid_model = model_builder.move_to(device)

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

reid_model.eval()

db = setup_database()

print("Getting all the frames from the DB...")
frames = db.get_all_frames()
total_frames = db.bboxes_col.count_documents({})

# cam > 
#   vehicle > 
#       frame id > 
#           [data['frame_number'], data['bounding_box'], data['features'], data['compressed_image'], data['shape'], data['timestamp']]
with tqdm(total=total_frames, desc="Updating features") as pbar:
  for id, chunk_dict in frames.items():
    for vid, vid_frames in chunk_dict.items():
      for frame_id, frame in vid_frames.items():
          decompressed_img = decompress_img(frame[3], frame[4], np.uint8)

          frame = Image.fromarray(decompressed_img).convert('RGB')  # Convert to PIL Image
          frame = transforms.val_transform(frame)             # Convert to torch.Tensor
          frame = frame.unsqueeze(0).float().to(device)

          features = reid_model.forward(frame, training=False)

          # update the frame with the features
          db.bboxes_col.update_one(
              {'_id': frame_id},
              {'$set': {'features': features.squeeze().detach().cpu().numpy().tolist()}}
          )
          pbar.update(1)