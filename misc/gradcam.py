import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
import os
from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add the parent directory of 'tracking' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import load_config_and_device, load_models_and_transformations

cfg, device = load_config_and_device()
_, model, _, transforms = load_models_and_transformations(device, cfg)
model = model.eval()

resnet_model = models.resnet50(pretrained=True).to(device).eval()

# LAYER
target_layers = [model.layer4[-1]]
target_layers_resnet = [resnet_model.layer4[-1]]
image_path = './data/test/TestImage.jpg'

# Caricare e preprocessare l'immagine
image = Image.open(image_path).convert('RGB')

input_tensor = transforms.val_transform(image).unsqueeze(0)  # Aggiungere dimensione batch
input_tensor = input_tensor.to(next(model.parameters()).device)

# We have to specify the target we want to generate the CAM for.
targets = [ClassifierOutputTarget(576)]

# Construct the CAM object once, and then re-use it on many images.
with GradCAM(model=model, target_layers=target_layers) as cam:
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    # resize the image to the shape of grayscale_cam
    image2show = image.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]), Image.Resampling.LANCZOS)

    # convert the image into a numpy array
    image2show = np.asarray(image2show, np.float32) / 255.0

    visualization_trained = show_cam_on_image(image2show, grayscale_cam, use_rgb=True)

with GradCAM(model=resnet_model, target_layers=target_layers_resnet) as cam:
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    # resize the image to the shape of grayscale_cam
    image2show = image.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]), Image.Resampling.LANCZOS)

    # convert the image into a numpy array
    image2show = np.asarray(image2show, np.float32) / 255.0

    visualization_resnet = show_cam_on_image(image2show, grayscale_cam, use_rgb=True)

# Visualizzare immagine originale e heatmap sovrapposta
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image2show)  # Sovrapporre heatmap con trasparenza
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(visualization_resnet)  # Sovrapporre heatmap con trasparenza
plt.axis('off')
plt.title('ResNet-IBN')

plt.subplot(1, 3, 3)
plt.imshow(visualization_trained)  # Sovrapporre heatmap con trasparenza
plt.axis('off')
plt.title('ResNet-IBN (w/ Tricks)')

plt.show()
