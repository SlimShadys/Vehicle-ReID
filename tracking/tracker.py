from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import ast

class DeepSortTracker(): 
    def __init__(self, config):
        # Extract configs
        self.config = config['tracker']

        # Set various parameters
        self.max_age = self.config['max_age'] # Maximum number of frames to keep a track alive without new detections. Default is 30
        self.n_init = self.config['n_init'] # Minimum number of detections needed to start a new track. Default is 3
        self.nms_max_overlap = self.config['nms_max_overlap'] # Maximum overlap between bounding boxes allowed for non-maximal supression (NMS).
                                                              # If two bounding boxes overlap by more than this value, the one with the lower confidence score is suppressed. Defaults to 1.0.
        self.max_iou_distance = self.config['max_iou_distance'] # Maximum IOU distance allowed for matching detections to existing tracks. Default is 0.7
        self.max_cosine_distance = self.config['max_cosine_distance']   # Maximum cosine distance allowed for matching detections to existing tracks. 
                                                                        # If the cosine distance between the detection's feature vector and the track's feature vector is higher than this value, 
                                                                        # the detection is not matched to the track. Defaults to 0.2
        self.nn_budget = self.config['nn_budget'] # Maximum number of nearest neighbors to look for when performing data association. Defaults to 100
        self.override_track_class = self.config['override_track_class'] # Whether to override the class of the detected objects with the class of the tracked object. Defaults to False
        self.embedder = self.config['embedder'] # The name of the feature extraction model to use. The options are "mobilenet" or "efficientnet". Defaults to "mobilenet".
        self.half = self.config['half'] # Whether to use half precision for feature extraction. If set to True, the feature extraction model will use half precision. Defaults to False.
        self.bgr = self.config['bgr'] # Whether to use BGR color format for images. If set to False, RGB format will be used. Defaults to True.
        self.embedder_gpu = self.config['embedder_gpu'] # Whether to use GPU for feature extraction. If set to False, CPU will be used. Defaults to True.
        self.embedder_model_name = self.config['embedder_model_name'] # The name of the feature extraction model to use. The options are "mobilenet" or "efficientnet". Defaults to "mobilenet".
        self.embedder_wts = self.config['embedder_wts'] # The path to the weights file for the feature extraction model. Defaults to None.
        self.polygon = self.config['polygon'] # The polygon to use for the object detection. If set to None, the entire frame will be used. Defaults to None.
        self.today = self.config['today'] # The current date. Defaults to None.

        # Set Tracker visualization parameters
        self.display_tracks = self.config['disp_tracks']
        self.disp_obj_track_box = self.config['disp_obj_track_box']
        self.obj_track_color = ast.literal_eval(self.config['obj_track_color'])
        self.obj_track_box_color = ast.literal_eval(self.config['obj_track_box_color'])

        # Initialize the DeepSort object tracker
        self.object_tracker = DeepSort(
            max_iou_distance=self.max_iou_distance, 
            max_age=self.max_age, 
            n_init=self.n_init, 
            nms_max_overlap=self.nms_max_overlap,
            max_cosine_distance=self.max_cosine_distance,
            nn_budget=self.nn_budget,
            override_track_class=self.override_track_class,
            embedder=self.embedder,
            half=self.half,
            bgr=self.bgr,
            embedder_gpu=self.embedder_gpu,
            embedder_model_name=self.embedder_model_name,
            embedder_wts=self.embedder_wts,
            polygon=self.polygon,
            today=self.today
        )
        
    def display_track(self, track_history, tracks_current, img):
        for track in tracks_current:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            
            # Retrieve the current track location (i.e - center of the bounding box) and bounding box
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Retrieve the previous center location, if available
            prev_centers = track_history.get(track_id ,[])
            prev_centers.append(bbox_center)
            track_history[track_id] = prev_centers
            
            # Draw the track line, if there is a previous center location
            if prev_centers is not None and self.display_tracks == True:
                points = np.array(prev_centers, np.int32)
                cv2.polylines(img, [points], False, self.obj_track_color, 2)

            if self.disp_obj_track_box == True: 
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.obj_track_box_color, 1)
                cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.obj_track_box_color, 1)

            