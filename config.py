# NOTE n.1 : If SAMPLER_TYPE "random" or "balanced" is selected, BATCH_SIZE must be divisible by NUM_INSTANCES.
# NOTE n.2 : If SAMPLER_TYPE "random" or "balanced" is selected, LOSS_TYPE must be "TripletLoss".
# NOTE n.3 : If USE_RPTM is True, LOSS_TYPE must be "TripletLossRPTM".

from yacs.config import CfgNode as ConfigNode

# ============================================================================================= #
#                                         CONFIGURATIONS                                        #
# ============================================================================================= #
_C = ConfigNode()

# ============================================================================================= #
#                                      MISC CONFIGURATIONS                                      #
# ============================================================================================= #
_C.MISC = ConfigNode()
_C.MISC.SEED = 2047315
_C.MISC.DEVICE = 'cuda'
_C.MISC.GPU_ID = 0
_C.MISC.USE_AMP = False
_C.MISC.OUTPUT_DIR = './reid/results/Test-50'

# ============================================================================================= #
#                                      ReID CONFIGURATIONS                                      #
# ============================================================================================= #
_C.REID = ConfigNode()

# ========> DATASET CONFIGS <========
_C.REID.DATASET = ConfigNode()
_C.REID.DATASET.DATA_PATH       = "./data"
_C.REID.DATASET.DATASET_NAME    = "veri_776" # veri_776 / veri_wild / vehicle_id / vru
_C.REID.DATASET.DATASET_SIZE    = "small" # small / medium / large
_C.REID.DATASET.SAMPLER_TYPE    = "random" # random / balanced / None
_C.REID.DATASET.NUM_INSTANCES   = 6 # Only to be adjusted if sampler_type is either "random" or "balanced"

# ========> MODEL CONFIGS <========
_C.REID.MODEL = ConfigNode()
_C.REID.MODEL.NAME                  = 'resnet50_ibn_a' # "resnet50" / "resnet101" / "resnet50_ibn_a" / "resnet101_ibn_a"
_C.REID.MODEL.PRETRAINED            = True
_C.REID.MODEL.USE_GEM               = False
_C.REID.MODEL.USE_STRIDE            = True
_C.REID.MODEL.USE_BOTTLENECK        = True
# -- Color Model Configuration
_C.REID.COLOR_MODEL                 = ConfigNode()
_C.REID.COLOR_MODEL.NAME            = 'efficientnet-b5' # "svm" / "efficientnet-b3" / "efficientnet-b5" (our implementation)
#_C.REID.COLOR_MODEL.PRETRAINED_PATH = "./tracking/car-color-classifier/Models/svm_classifier_v3.pkl"                       # SVM Pre-trained Model
_C.REID.COLOR_MODEL.PRETRAINED_PATH = "./tracking/car-color-classifier/Models/efficientnet-b5_loss-0.1856_acc-94.7458.pt"   # EfficientNet Pre-trained Model

# ========> AUGMENTATION CONFIGS <========
_C.REID.AUGMENTATION = ConfigNode()
_C.REID.AUGMENTATION.HEIGHT                         = 320
_C.REID.AUGMENTATION.WIDTH                          = 320
_C.REID.AUGMENTATION.RANDOM_CROP                    = [320, 320]
_C.REID.AUGMENTATION.RANDOM_HORIZONTAL_FLIP_PROB    = 0.5
_C.REID.AUGMENTATION.JITTER_BRIGHTNESS              = 0.2
_C.REID.AUGMENTATION.JITTER_CONTRAST                = 0.15
_C.REID.AUGMENTATION.JITTER_SATURATION              = 0.0
_C.REID.AUGMENTATION.JITTER_HUE                     = 0.0
_C.REID.AUGMENTATION.COLOR_AUGMENTATION             = True
_C.REID.AUGMENTATION.PADDING                        = 0
_C.REID.AUGMENTATION.NORMALIZE_MEAN                 = None # [0.485, 0.456, 0.406]
_C.REID.AUGMENTATION.NORMALIZE_STD                  = None # [0.229, 0.224, 0.225]
_C.REID.AUGMENTATION.RANDOM_ERASING_PROB            = 0.5

# ========> LOSS CONFIGS <========
_C.REID.LOSS = ConfigNode()
_C.REID.LOSS.TYPE               = 'TripletLoss' # "TripletLoss" / "TripletLossRPTM" / "SupConLoss" / "SupConLoss_Pytorch" / "TripletMarginLoss_Pytorch"
# -- RPTM Configuration
_C.REID.LOSS.USE_RPTM           = [False, "mean"] # [True/False, "mean"/"max"/"min"]
# -- TripetLoss Configuration
_C.REID.LOSS.MARGIN             = 1.00 
# -- Label Smoothing Configuration
_C.REID.LOSS.LABEL_SMOOTHING    = 0.00
# -- MALW Configuration
_C.REID.LOSS.APPLY_MALW         = False
_C.REID.LOSS.ALPHA              = 0.9
_C.REID.LOSS.K                  = 2000

# ========> TRAINING CONFIGS <========
_C.REID.TRAINING = ConfigNode()
_C.REID.TRAINING.EPOCHS = 160
_C.REID.TRAINING.BATCH_SIZE = 36
_C.REID.TRAINING.NUM_WORKERS = 0
# -- Optimizer Configuration
_C.REID.TRAINING.OPTIMIZER = "adam" # adam / sgd
_C.REID.TRAINING.LEARNING_RATE = 1.5e-4
_C.REID.TRAINING.BIAS_LR_FACTOR = 2
_C.REID.TRAINING.WEIGHT_DECAY = 0.0005
_C.REID.TRAINING.WEIGHT_DECAY_BIAS = 0.0001
# -- Scheduler Configuration (WarmupDecayLR or MultiStepLR)
_C.REID.TRAINING.USE_WARMUP = True
_C.REID.TRAINING.STEPS = [20, 30, 45, 60, 75, 90, 105, 120, 135, 140]
_C.REID.TRAINING.GAMMA = 0.6
_C.REID.TRAINING.WARMUP_EPOCHS = 10
_C.REID.TRAINING.DECAY_METHOD = "cosine" # "linear" / "smooth" / "cosine"
_C.REID.TRAINING.COSINE_POWER = 1.00 # Only to be adjusted if decay_method is "cosine"
_C.REID.TRAINING.MIN_LR = 1.0e-6
# -- Logging Configuration
_C.REID.TRAINING.LOG_INTERVAL = 100
# -- Loading Checkpoint
_C.REID.TRAINING.LOAD_CHECKPOINT = False # Could be also a path to a checkpoint

# ========> VALIDATION CONFIGS <========
_C.REID.VALIDATION = ConfigNode()
_C.REID.VALIDATION.BATCH_SIZE = 16
_C.REID.VALIDATION.VAL_INTERVAL = 1
_C.REID.VALIDATION.RE_RANKING = True
_C.REID.VALIDATION.VISUALIZE_RANKS = False

# ========> TEST CONFIGS <========
_C.REID.TEST = ConfigNode()
_C.REID.TEST.TESTING = False # Always set to False (except in config_test.yml)
_C.REID.TEST.RUN_REID_METRICS = False
_C.REID.TEST.RUN_COLOR_METRICS = False
_C.REID.TEST.PRINT_CONFUSION_MATRIX = True # Only available if run_color_metrics is True
_C.REID.TEST.NORMALIZE_EMBEDDINGS = True
_C.REID.TEST.MODEL_VAL_PATH = "./reid/results/Test-14/model_ep-158_loss-0.0275.pth"
_C.REID.TEST.PATH_IMG_1 = "data/test/honda_f.jpg"
_C.REID.TEST.PATH_IMG_2 = "data/test/honda_f_2.jpg"
_C.REID.TEST.STACK_IMAGES = True # Stack all the images to create a NxN matrix of similarity scores. Only available if run_reid_metrics is False
_C.REID.TEST.SIMILARITY_ALGORITHM   = "cosine"  # "euclidean" / "cosine"
_C.REID.TEST.SIMILARITY_METHOD      = "individual"    # "individual" / "mean"

# ============================================================================================= #
#                                    TRACKING CONFIGURATIONS                                    #
# ============================================================================================= #

# ========> MISC TRACKING CONFIGS <========
_C.TRACKING = ConfigNode()

# ========> YOLO MODEL CONFIGS <========
_C.TRACKING.MODEL = ConfigNode()
_C.TRACKING.MODEL.YOLO_MODEL_NAME = 'yolov11x'
_C.TRACKING.MODEL.TRACKED_CLASS = [(1, 'bicycle'), (2, 'car'), (3, 'motorcycle'), (5, 'bus'), (6, 'train'), (7, 'truck')]
_C.TRACKING.MODEL.CONFIDENCE_THRESHOLD = 0.50           # Minimum confidence threshold for detections
_C.TRACKING.MODEL.YOLO_IOU_THRESHOLD = 0.90             # Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS)
_C.TRACKING.MODEL.YOLO_IMAGE_SIZE = (640, 1216)         # Defines the image size for inference | Either a single integer or a tuple (height, width)
_C.TRACKING.MODEL.USE_AGNOSTIC_NMS = True               # If True, NMS is applied independently to each class
_C.TRACKING.MODEL.YOLO_TRACKER = 'custom_tracker.yaml' # YOLO Tracker Configuration File (custom_tracker.yaml / botsort.yaml / bytetrack.yaml)

# ========> FILTERING CONFIGS <========
_C.TRACKING.STATIONARY_THRESHOLD = 0.01
_C.TRACKING.MIN_STATIONARY_FRAMES = 1000
_C.TRACKING.MIN_BOX_SIZE = 60

# Output Tracking Configs
_C.TRACKING.BOUNDING_BOXES_OUTPUT_PATH = './tracking/bounding_boxes/'   # Necessary just to Debugging purposes
_C.TRACKING.OUTPUT_PATH = './tracking/output/'                          # Necessary just to Debugging purposes
_C.TRACKING.SAVE_VIDEO = False                                          # Saves Tracking Video in the folder specified above (Debugging purposes)
_C.TRACKING.SAVE_BOUNDING_BOXES = True                                  # Saves BBoxes in a folder (Debugging purposes)
_C.TRACKING.DISP_INFOS = True                                           # Display information on top-left corner of the video

# ============================================================================================= #
#                                    DATABASE CONFIGURATIONS                                    #
# ============================================================================================= #
_C.DB = ConfigNode()
_C.DB.URL = "mongodb://localhost:27017/"
_C.DB.NAME = "ReID"
_C.DB.VEHICLES_COL = "Vehicles"
_C.DB.CAMERAS_COL = "Cameras"
_C.DB.TRAJECTORIES_COL = "Trajectories"
_C.DB.BBOXES_COL = "BBoxes"
_C.DB.VERBOSE = False

# ================================================================================================== #
#                                    FULL PIPELINE CONFIGURATIONS                                    #
# ================================================================================================== #
_C.TRACKING.VIDEO_PATH      = './tracking/video/output_camera2.mp4'                     # Path to the video file
_C.MISC.TARGET_COLOR        = ['black', 'blue']                                         # Target colors for the Re-ID filtering process
_C.MISC.RUN_PIPELINE        = True                                                      # If True, the pipeline will run the full process of tracking, Re-ID and DB insertion
_C.MISC.RUN_UNIFICATION     = False                                                     # If True, the pipeline will ONLY run the unification process of the database
_C.DB.CLEAN_DB              = True                                                      # If True, the database will be cleaned before inserting new data
_C.TRACKING.USE_ROI_MASK    = True                                                      # If True, the pipeline will use a ROI mask to track the vehicles
_C.TRACKING.ROI_MASK_PATH   = './tracking/video/roi_masks/output_camera2_roi_mask.png'  # Path to the ROI mask. Leave empty for creating a new one. USE_ROI_MASK must be True.