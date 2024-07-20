import cv2
import torch

class YOLOv5():
    def __init__(self, config, device='cpu'):
        # Extract configs
        self.config = config['detector']

        # Set the model name, downscale factor, confidence threshold, tracked class and display object detection box
        self.model_name = self.config['yolo_model_name']
        self.downscale_factor = self.config['downscale_factor']         # Reduce the resolution of the input frame by this factor to speed up object detection process
        self.confidence_threshold = self.config['confidence_threshold'] # Minimum theshold for the detection bounding box to be displayed
        self.tracked_class = self.config['tracked_class']
        self.disp_obj_detect_box = self.config['disp_obj_detect_box']

        if self.model_name:
            self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.device = device
        self.model.to(self.device) # Set the model to the device
        self.classes = self.model.names

    def run(self, frame):
        if(self.downscale_factor != 1):
            frame = cv2.resize(frame, (int(frame.shape[1]/self.downscale_factor), int(frame.shape[0]/self.downscale_factor)))

        # Perform object detection
        yolo_result = self.model(frame)

        # Extract the labels and bounding box coordinates
        labels = yolo_result.xyxyn[0][:,-1]
        bb_cord = yolo_result.xyxyn[0][:,:-1]
        
        return (labels, bb_cord)

    def class_to_label(self, x):
        return self.classes[int(x)]
        
    def extract_detections(self, results, frame, height, width):
        labels, bb_cordinates = results  # Extract labels and bounding box coordinates
        detections = []         # Empty list to store the detections later 
        class_count = 0         # Initialize class count for the frame 
        num_objects = len(labels)   #extract the number of objects detected
        x_shape, y_shape = width, height

        for object_index in range(num_objects):
            row = bb_cordinates[object_index]
            conf_val = float(row[4].item()).__round__(2)

            if conf_val >= self.confidence_threshold:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if self.class_to_label(labels[object_index]) in self.tracked_class:
                    
                    if self.disp_obj_detect_box: 
                        self.plot_boxes(x1, y1, x2, y2, frame)

                    # x_center = x1 + ((x2 - x1) / 2)
                    # y_center = y1 + ((y2 - y1) / 2)

                    class_count += 1
                    
                    # We structure the detections in this way because we want the bbs expected
                    # to be a list of detections in the tracker, each in tuples of
                    # ([left, top, w, h], confidence, detection_class)
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], conf_val, self.tracked_class))
                    
        return detections, class_count
    
    def plot_boxes(self, x1, y1, x2, y2, frame):  
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
