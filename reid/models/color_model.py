import pickle
import torch

# EfficientNet model
class EfficientNet():
    def __init__(self, configs=None, device='cuda'):
        # Variables
        self.configs = configs
        self.device = device
        self.model = torch.load(self.configs.PRETRAINED_PATH)
        self.model = self.model.to(self.device)
        self.mappings = {
            0: 'yellow',
            1: 'orange',
            2: 'green',
            3: 'gray',
            4: 'red',
            5: 'blue',
            6: 'white',
            7: 'golden',
            8: 'brown',
            9: 'black',
            10: 'dark gray',
            11: 'purple',
            12: 'cyan'
        }

    def predict(self, img):
        preds = []
        color_predictions = self.model(img)

        # Convert from logits to probs
        color_predictions = torch.nn.functional.softmax(color_predictions, dim=1)
        for _, color in enumerate(color_predictions):
                color_prediction = self.mappings[torch.argmax(color).item()]
                prob = str(color[torch.argmax(color)].item())
                preds.append({"color": color_prediction, "prob": prob})
        return preds
    
    def to(self, device):
        self.model = self.model.to(device) # Move the model to the device
        return self
    
class SVM():
    def __init__(self, configs=None):
        self.configs = configs
        self.model = pickle.load(open(self.configs.PRETRAINED_PATH, 'rb'))
        self.colors = ['yellow','orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black', 'dark gray', 'purple', 'cyan']

    def predict(self, embedding):
        preds = []
        color_preds = self.model.predict(embedding)
        for color in color_preds:
            preds.append(self.colors[color])
        return preds