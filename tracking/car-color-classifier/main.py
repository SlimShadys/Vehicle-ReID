# import the necessary packages
import random
import time

import cv2
import models
import numpy as np
import tensorflow as tf
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print("Correctly set the seed to:", seed)

def main(config, seed):
    
    # Set the seed for reproducibility
    set_seed(seed)
        
    # construct the argument parse and parse the arguments
    car_classifier = models.CarClassifier(config)

    # load our input image and grab its spatial dimensions
    image_name = config.get('misc', {}).get('image', None)
    image = cv2.imread(image_name) # 416, 372, 3 in uint8

    start = time.time()

    # Run inference
    result = car_classifier.color_classifier.predict(image)
    #result_model = car_classifier.type_classifier.predict(image)

    end = time.time()

    # show timing information on YOLO
    print("[INFO] NN took {:.6f} seconds".format(end - start))

    print(result)
    #print(result_model)

# Usage: python main.py <path_to_config.yml>
if __name__ == '__main__':
    config_file = 'car-color-classifier/config.yml'

    # Parameters from config.yml file
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)['configs']

    # Get the seed from the config
    # Default to 2047315 if not specified
    seed = config.get('misc', {}).get('seed', 2047315)

    # Run the main function with the seed
    main(config, seed)