from ultralytics import YOLO
import yacs.config

def load_yolo(config: yacs.config.CfgNode):
    yolo_name = config.MODEL.YOLO_MODEL_NAME
    model = YOLO(yolo_name + '.pt') # Load the YOLO model
    
    print("Successfully loaded YOLO model:", yolo_name)
    
    return model