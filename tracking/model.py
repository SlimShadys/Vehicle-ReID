import yacs.config
from ultralytics import YOLO

def load_yolo(config: yacs.config.CfgNode):
    yolo_name = config.DETECTOR.MODEL.NAME
    model = YOLO(yolo_name + '.pt') # Load the YOLO model
    
    print("Successfully loaded YOLO model:", yolo_name)
    
    return model