# Copyright © 2019-2020 by Spectrico
# Licensed under the MIT License

import cv2
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf  # TensorFlow 2.0
from utils import load_graph, load_labels, resizeAndPad

class TypeClassifier():
    def __init__(self, configs=None, topK=3):
    
        # Variables
        self.configs = configs
        self.topK = topK
        
        self.classifier_input_size = self.configs['input_size']
        self.input_layer = self.configs['input_layer']
        self.output_layer = self.configs['output_layer']
        self.model_file = self.configs['model_file']
        self.label_file = self.configs['label_file']

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
        img /= 127.5
        img -= 1.

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
        
        self.classifier_input_size = self.configs['input_size']
        self.input_layer = self.configs['input_layer']
        self.output_layer = self.configs['output_layer']
        self.model_file = self.configs['model_file']
        self.label_file = self.configs['label_file']

        self.graph = load_graph(self.model_file)
        self.labels = load_labels(self.label_file)

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

        self.sess = tf.Session(graph=self.graph)
        self.sess.graph.finalize()  # Graph is read-only after this statement.

    def predict(self, img):
        img = img[:, :, ::-1] # BGR to RGB
        img = cv2.resize(img, self.classifier_input_size) # 224, 224, 3

        # Add a forth dimension since Tensorflow expects a list of images
        img = np.expand_dims(img, axis=0)

        # Scale the input image to the range used in the trained network
        img = img.astype(np.float32) # 1, 224, 224, 3
        img /= 127.5
        img -= 1.

        results = self.sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: img})
        results = np.squeeze(results)

        top_indices = results.argsort()[-self.topK:][::-1]
        classes = []
        for ix in top_indices:
            classes.append({"color": self.labels[ix], "prob": str(results[ix])})
        return(classes)

class CarClassifier():
    def __init__(self, configs=None):
        # uncomment the next 3 lines if you want to use CPU instead of GPU
        #import os
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Variables
        self.configs = configs

        # Top K results to show
        self.topK = self.configs.get('topK', 3)

        # Instantiate the classifiers        
        self.color_classifier = ColorClassifier(self.configs['color_classifier'], topK=self.topK)
        #self.type_classifier = TypeClassifier(self.configs['type_classifier'], topK=self.topK)