# Copyright © 2019-2020 by Spectrico
# Licensed under the MIT License

import cv2
import numpy as np
import tensorflow.compat.v1 as tf  # For TensorFlow 2.0
from misc.utils import load_graph, load_labels, resizeAndPad
import matplotlib.pyplot as plt

class TypeClassifier():
    def __init__(self, configs=None, topK=3):
    
        # Variables
        self.configs = configs
        self.topK = topK
        
        self.classifier_input_size = self.configs.INPUT_SIZE
        self.input_layer = self.configs.INPUT_LAYER
        self.output_layer = self.configs.OUTPUT_LAYER
        self.model_file = self.configs.MODEL_FILE
        self.label_file = self.configs.LABEL_FILE

        self.graph = load_graph(self.model_file)
        self.labels = load_labels(self.label_file)

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

        self.sess = tf.Session(graph=self.graph)
        self.sess.graph.finalize()  # Graph is read-only after this statement.

    def predict(self, img):
        img = img[:, :, ::-1]
        img = resizeAndPad(img, self.classifier_input_size)

        # Add a forth dimension since Tensorflow expects a list of images
        img = np.expand_dims(img, axis=0)

        # Scale the input image to the range used in the trained network
        img = img.astype(np.float32)
        img = img / 255.0

        results = self.sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: img})
        results = np.squeeze(results)

        top_indices = results.argsort()[-self.topK:][::-1]
        classes = []
        for ix in top_indices:
            make_model = self.labels[ix].split('\t')
            classes.append({"make": make_model[0], "model": make_model[1], "prob": str(results[ix])})
        return(classes)

class ColorClassifier():
    def __init__(self, configs=None, topK=3):
    
        # Variables
        self.configs = configs
        self.topK = topK
        
        self.classifier_input_size = self.configs.INPUT_SIZE
        self.input_layer = self.configs.INPUT_LAYER
        self.output_layer = self.configs.OUTPUT_LAYER
        self.model_file = self.configs.MODEL_FILE
        self.label_file = self.configs.LABEL_FILE

        self.graph = load_graph(self.model_file)
        self.labels = load_labels(self.label_file)

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

        self.sess = tf.Session(graph=self.graph)
        self.sess.graph.finalize()  # Graph is read-only after this statement.

    def predict(self, img_batch):
        # Converted flag to determine if the input was a tensor or a numpy array
        #    - True: input was a tensor (likely a color metric test) – no image processing needed.
        #    - False: input was a numpy array (likely from a full pipeline procedure) – image processing required.
        converted = False
        
        # Create a new numpy array to accomodate the resized images
        img_batch_resized = np.zeros((len(img_batch), self.classifier_input_size[0], self.classifier_input_size[1], 3))

        if type(img_batch) is not np.ndarray:
            img_batch = img_batch.detach().cpu().numpy()
            img_batch = np.transpose(img_batch, (0, 2, 3, 1))
            converted = True
            
        for i, img in enumerate(img_batch):
            img = cv2.resize(img, self.classifier_input_size)       # Resize the image
            if not converted:
                img = img[:, :, ::-1]                               # Reverse the channels from RGB to BGR
                img = img / 255.0                                   # Normalize the image to the range [0, 1]
                img = img.astype(np.float32)                        # Convert the image to float32
            img_batch_resized[i] = img

        # Run the prediction for the entire batch
        results = self.sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: img_batch_resized})

        # Process the results for each image in the batch
        batch_classes = []
        for result in results:
            result = np.squeeze(result)
            top_indices = result.argsort()[-self.topK:][::-1]
            classes = []
            for ix in top_indices:
                classes.append({"color": self.labels[ix], "prob": str(result[ix])})
            batch_classes.append(classes)
        
        return batch_classes

class CarClassifier():
    def __init__(self, configs=None):
        # Uncomment if you want to run the model on CPU
        #import os
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Variables
        self.configs = configs

        # Top K results to show
        self.topK = self.configs.TOP_K

        # Instantiate the classifiers        
        self.color_classifier = ColorClassifier(self.configs, topK=self.topK)
        #self.type_classifier = TypeClassifier(self.configs, topK=self.topK)