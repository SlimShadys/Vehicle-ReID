# NOTE n.1 : If SAMPLER_TYPE "random" or "balanced" is selected, BATCH_SIZE must be divisible by NUM_INSTANCES.
# NOTE n.2 : If SAMPLER_TYPE "random" or "balanced" is selected, LOSS_TYPE must be "TripletLoss".
# NOTE n.3 : If USE_RPTM is True, LOSS_TYPE must be "TripletLossRPTM".
# NOTE n.4 : If PADDING_MODE is:
#               - "aspect_ratio" : Make sure to NOT set the RandomCrop() transformation, as we do process
#                                  the images one by one due to size issues with BatchNorm2d layer.
#               - "centered"     : We process the images in batches directly as they do have the same size.
#                                  Padding is automatically applied to the images.
#            In general, if Resize is an integer, the aspect ratio is kept either way.
#
#  -======================================-
# | USEFUL INFO:                           |
# | VRIC in total has 60.430 images        |
# | AICity in total has 85.058 images      |
# | AICity Sim in total has 192.150 images |
#  -======================================-

from yacs.config import CfgNode as ConfigNode

# ============================================================================================= #
#                                         CONFIGURATIONS                                        #
# ============================================================================================= #
_C = ConfigNode()
_C.DB                   = ConfigNode()
_C.DETECTOR             = ConfigNode()
_C.DETECTOR.MODEL       = ConfigNode()
_C.MISC                 = ConfigNode()
_C.MTMC                 = ConfigNode()
_C.REID                 = ConfigNode()
_C.REID.AUGMENTATION    = ConfigNode()
_C.REID.COLOR_MODEL     = ConfigNode()
_C.REID.DATASET         = ConfigNode()
_C.REID.LOSS            = ConfigNode()
_C.REID.MODEL           = ConfigNode()
_C.REID.TEST            = ConfigNode()
_C.REID.TRAINING        = ConfigNode()
_C.REID.VALIDATION      = ConfigNode()
_C.TRACKER              = ConfigNode()
_C.TRACKER.MODEL        = ConfigNode()
_C.TRACKING             = ConfigNode()
_C.YOLO                 = ConfigNode()
_C.METRICS              = ConfigNode()

# ============================================================================================= #
#                                      MISC CONFIGURATIONS                                      #
# ============================================================================================= #
_C.MISC.SEED = 2047315
_C.MISC.DEVICE = 'cuda' # 'cpu' / 'cuda'
_C.MISC.GPU_ID = 0
_C.MISC.USE_AMP = False

# ============================================================================================= #
#                                      ReID CONFIGURATIONS                                      #
# ============================================================================================= #

# Output ReID Directory
_C.REID.OUTPUT_DIR = './reid/results/Test-31'

# ========> DATASET CONFIGS <========
_C.REID.DATASET.DATA_PATH       = "./data"
_C.REID.DATASET.DATASET_NAME    = 'ai_city' # ai_city / ai_city_mix / ai_city_sim / vehicle_id / veri_776 / veri_wild / vric / vru
_C.REID.DATASET.SPLITTINGS      = [('vric', 0.2481), ('ai_city', 0.3528), ('ai_city_sim', 0.026)] # Only to be adjusted if dataset_name is a list (CombinedDataset)
_C.REID.DATASET.DATASET_SIZE    = "small" # small / medium / large
_C.REID.DATASET.SAMPLER_TYPE    = "random" # random / balanced / None
_C.REID.DATASET.NUM_INSTANCES   = 6 # Only to be adjusted if sampler_type is either "random" or "balanced". Look at the NOTE n.1

# ========> MODEL CONFIGS <========
_C.REID.MODEL.NAME                  = 'resnet50_ibn_a' # "resnet50" / "resnet101" / "resnet50_ibn_a" / "resnet101_ibn_a"
_C.REID.MODEL.PRETRAINED            = True
_C.REID.MODEL.RUN_FINETUNING        = False  # If True, the weights from _C.REID.TEST.MODEL_VAL_PATH are loaded and the model is fine-tuned
_C.REID.MODEL.USE_GEM               = False
_C.REID.MODEL.USE_STRIDE            = True
_C.REID.MODEL.USE_BOTTLENECK        = True
# -- Color Model Configuration
_C.REID.COLOR_MODEL.NAME            = 'efficientnet-b5' # "svm" / "efficientnet-b3" / "efficientnet-b5" (our implementation)
#_C.REID.COLOR_MODEL.PRETRAINED_PATH = "./tracking/car-color-classifier/Models/svm_classifier_v3.pkl"                       # SVM Pre-trained Model
_C.REID.COLOR_MODEL.PRETRAINED_PATH = "./tracking/car-color-classifier/Models/efficientnet-b5_loss-0.1856_acc-94.7458.pt"   # EfficientNet Pre-trained Model

# ========> AUGMENTATION CONFIGS <========
_C.REID.AUGMENTATION.PADDING_MODE                   = "centered" # "centered" / "aspect_ratio" | For training! MANDATORY -> Look at NOTE n.4
_C.REID.AUGMENTATION.RESIZE                         = 320 # 320 keeps image aspect ratio. (320, 320) doesn't.
_C.REID.AUGMENTATION.RANDOM_CROP                    = 0   # WATCH OUT: Activating random crop with a simple integer number, will resize the image to a square.
_C.REID.AUGMENTATION.RANDOM_HORIZONTAL_FLIP_PROB    = 0.5
_C.REID.AUGMENTATION.JITTER_BRIGHTNESS              = 0.0
_C.REID.AUGMENTATION.JITTER_CONTRAST                = 0.0
_C.REID.AUGMENTATION.JITTER_SATURATION              = 0.0
_C.REID.AUGMENTATION.JITTER_HUE                     = 0.0
_C.REID.AUGMENTATION.COLOR_AUGMENTATION             = True
_C.REID.AUGMENTATION.PADDING                        = 10 # 0 for deactivating padding
_C.REID.AUGMENTATION.NORMALIZE_MEAN                 = None # [0.485, 0.456, 0.406]
_C.REID.AUGMENTATION.NORMALIZE_STD                  = None # [0.229, 0.224, 0.225]
_C.REID.AUGMENTATION.RANDOM_ERASING_PROB            = 0.5

# ========> LOSS CONFIGS <========
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
_C.REID.TRAINING.EPOCHS = 160
_C.REID.TRAINING.BATCH_SIZE = 12
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
_C.REID.VALIDATION.BATCH_SIZE = 16
_C.REID.VALIDATION.VAL_INTERVAL = 1
_C.REID.VALIDATION.RE_RANKING = True
_C.REID.VALIDATION.VISUALIZE_RANKS = False

# ========> TEST CONFIGS <========
_C.REID.TEST.TESTING = False # Always set to False (except in config_test.yml)
_C.REID.TEST.RUN_REID_METRICS = False
_C.REID.TEST.RUN_COLOR_METRICS = False
_C.REID.TEST.PRINT_CONFUSION_MATRIX = True # Only available if run_color_metrics is True
_C.REID.TEST.NORMALIZE_EMBEDDINGS = True
_C.REID.TEST.MODEL_VAL_PATH = "./reid/results/Test-31/model_ep-156_loss-0.0578.pth"
_C.REID.TEST.PATH_IMG_1 = "data/test/honda_f.jpg"
_C.REID.TEST.PATH_IMG_2 = "data/test/honda_f_2.jpg"
_C.REID.TEST.STACK_IMAGES = True # Stack all the images to create a NxN matrix of similarity scores. Only available if run_reid_metrics is False

# ============================================================================================= #
#                                    TRACKING CONFIGURATIONS                                    #
# ============================================================================================= #

# ========> GENERAL YOLO CONFIGS <========
_C.YOLO.TRACKED_CLASS = [(1, 'bicycle'), (2, 'car'), (3, 'motorcycle'), (5, 'bus'), (6, 'train'), (7, 'truck')]
_C.YOLO.CONFIDENCE_THRESHOLD = 0.20           # Minimum confidence threshold for detections
_C.YOLO.YOLO_IOU_THRESHOLD = 0.85             # Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS)
_C.YOLO.YOLO_IMAGE_SIZE = (640, 640)        # Defines the image size for inference | Either a single integer or a tuple (height, width)
_C.YOLO.USE_AGNOSTIC_NMS = True               # If True, NMS is applied independently to each class
_C.YOLO.VERBOSE = False

# ========> DETECTOR <========
_C.DETECTOR.MODEL.NAME = 'yolov11x' 

# ========> TRACKER <========
_C.TRACKER.MODEL.NAME = 'yolov11x'
_C.TRACKER.YOLO_TRACKER = './tracking/custom_tracker.yaml'       # YOLO Tracker Configuration File (./tracking/custom_tracker.yaml / botsort.yaml / bytetrack.yaml)
_C.TRACKER.PERSIST = True

# ========> FILTERING CONFIGS <========
_C.TRACKING.STATIONARY_THRESHOLD = 0.01
_C.TRACKING.MIN_STATIONARY_FRAMES = 1000
_C.TRACKING.MIN_BOX_SIZE = 60

# Output Tracking Configs
_C.TRACKING.BOUNDING_BOXES_OUTPUT_PATH = './tracking/bounding_boxes/'   # Necessary just to Debugging purposes
_C.TRACKING.OUTPUT_PATH = './tracking/output/'                          # Necessary just to Debugging purposes
_C.TRACKING.SAVE_VIDEO = False                                          # Saves Tracking Video in the folder specified above (Debugging purposes)
_C.TRACKING.SAVE_BOUNDING_BOXES = False                                  # Saves BBoxes in a folder (Debugging purposes)
_C.TRACKING.DISP_INFOS = True                                           # Display information on top-left corner of the video

# ============================================================================================= #
#                                    DATABASE CONFIGURATIONS                                    #
# ============================================================================================= #
_C.DB.URL = "mongodb://localhost:27017/"
_C.DB.NAME = "ReID"
_C.DB.VEHICLES_COL = "Vehicles"
_C.DB.CAMERAS_COL = "Cameras"
_C.DB.TRAJECTORIES_COL = "Trajectories"
_C.DB.BBOXES_COL = "BBoxes"
_C.DB.VERBOSE = False

# ============================================================================================= #
#                                     METRICS CONFIGURATIONS                                    #
#                                                                                               #
#       You should care about these only if you want to run MOTA metrics and check results      #
# ============================================================================================= #
# The prediction list must be populated only if we run 'python evaluate_AICity.py'
# If you evaluate a single camera  => Use a single MTSC file
# If you evaluate multiple cameras => Use multple MTMC files
#
# Normally, if you run the pipeline with _C.MISC.EVALUATE_METRICS = True, it will
# automatically populate this list with the MTMC files generated by the pipeline itself.
_C.METRICS.PREDICTIONS = [
        # "MTMC-predictions_camera-0_2024-12-13_14-41-05.txt", # cam006
        # "MTMC-predictions_camera-1_2024-12-13_14-41-06.txt", # cam007
        "MTSC-predictions_camera-2_2024-12-16_10-11-17.txt", # cam008
        # "MTMC-predictions_camera-3_2024-12-13_14-41-07.txt", # cam009
    ]
_C.METRICS.MIN_IOU = 0.30
_C.METRICS.IGNORE_FP = True
_C.METRICS.DROP_SINGLE_CAM_TRACKS = True

# ============================================================================================= #
#                                    PIPELINE CONFIGURATIONS                                    #
# ============================================================================================= #
_C.MISC.TEST_NAME                       = 'Test-34'                                                             # Name of the test
_C.TRACKING.VIDEO_PATH                  = './data/AICity22_Track1_MTMC_Tracking/validation/S02/c006/vdo.avi'    # Path to the video file or folder containing the video frames
_C.TRACKING.USE_ROI_MASK                = True                                                                  # If True, the pipeline will use a ROI mask to track the vehicles
_C.TRACKING.ROI_MASK_PATH               = './data/AICity22_Track1_MTMC_Tracking/validation/S02/c006/roi.jpg'    # Path to the ROI mask. Leave empty for creating a new one. USE_ROI_MASK must be True.
_C.MISC.TARGET_PATH                     = ('./target/Target_2.png', 1)                                          # Path to the target file (PNG, Camera ID)
_C.MISC.TARGET_COLOR                    = ['']                                                                  # Target colors for the Re-ID filtering process | [''] / ['red'] / ['blue'] / ['red', 'blue'] / etc
_C.DB.USE_DB                            = True                                                                  # If True, the pipeline will insert the data into the database
_C.DB.CLEAN_DB                          = True                                                                  # If True, the database will be cleaned before inserting new data
_C.MISC.USE_FILTERS                     = False                                                                 # If True, the pipeline will use filters to remove unwanted vehicles (color, stationary and incomplete)
_C.TRACKING.SAMPLE_FRAMES               = False                                                                 # If True, the pipeline will sample maximum 50 frames to reduce the number of frames stored
_C.TRACKING.DELETE_MIN_FRAMES           = True                                                                  # If True, the pipeline will delete the frames after the tracking process
_C.MISC.RUN_UNIFICATION                 = True                                                                  # If True, the pipeline will ONLY run the unification process of the database
_C.MISC.RUN_PIPELINE                    = True                                                                  # If True, the pipeline will run the full process of tracking, Re-ID and DB insertion
_C.MTMC.SHOW_MTMC_IMAGES                = False                                                                 # If True, the pipeline will show the MTMC results in terms of plotting
_C.MISC.EVALUATE_METRICS                = True                                                                  # If True, the pipeline will run the MTMC metrics
_C.MISC.RUN_TARGET_SEARCH               = False                                                                 # If True, the pipeline will run the Target Search process
_C.MTMC.USE_PICKLE                      = True                                                                  # If True, the pipeline will use the pickle file to load the MTMC/MTSC data
_C.MTMC.PICKLE_GLOBAL_PATH              = './tracking/pickle_trajectories'                                      # Path to the pickle file containing the MTMC/MTSC data

# Similarity Algorithm and Methods
_C.REID.TEST.SIMILARITY_ALGORITHM       = "cosine"              # "euclidean"  / "cosine"
_C.REID.TEST.UNIFICATION_METHOD         = "area_avg"            # "mean" / "area_avg" / "individual"
_C.REID.TEST.SIMILARITY_METHOD          = "area_avg"            # "mean" / "area_avg"
_C.REID.TEST.SIMILARITY_THRESHOLD       = 0.50                  # Threshold for the similarity score
_C.REID.TEST.LINKAGE_METHOD             = "mean_feature"        # "mean_feature" / "average" / "single" / "complete"
_C.MTMC.TARGET_SEARCH_METHOD            = "cameralink_search"   # "cameralink_search" / "multitrack_search" / "singletrack_search"
