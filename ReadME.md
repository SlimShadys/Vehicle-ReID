# Vehicle Re-Identification
<p align="center">
  <img alt="Normal images" src="https://github.com/SlimShadys/Vehicle-ReID/blob/master/GIFs/LeftAngle.gif" width="45%"> 
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Density maps" src="https://github.com/SlimShadys/Vehicle-ReID/blob/master/GIFs/RightAngle.gif" width="45%">
</p>

## Install proper libraries
In order to install the proper libraries, please first install PyTorch. The following guidelines are made for Windows 10/11. Adjust accordingly.

### PyTorch
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Once everything has been installed, simply run the `requirements.txt` file with:
```python
pip install -r requirements.txt
```

## Configuration
The project has a dedicated `config.py` file where all the default parameters have been set, such as:

- <b>Misc</b>: Contains general settings such as the random seed, device configuration (CPU/GPU), and experimental features like Automatic Mixed Precision (AMP).
- <b>Dataset</b>: Defines the dataset-related configurations, including paths, dataset names, sizes, and sampling strategies. It also handles dataset splitting for combined datasets.
- <b>Model</b>: Configures the ReID model architecture, including backbone choices, pretraining options, and advanced features like GeM pooling, stride adjustments, and normalization layers.
- <b>Color Model</b>: Specifies the color classification model used alongside the ReID model, including its architecture and pretrained weights.
- <b>Augmentation</b>: Controls the data augmentation pipeline, including resizing, cropping, padding, color jitter, and normalization settings.
- <b>Loss</b>: Configures the loss functions used during training, including triplet loss variants, label smoothing, and advanced techniques like Relation Preserving Triplet Mining (RPTM) and Multi-Attribute Loss Weighting (MALW).
- <b>Training</b>: Manages the training process, including epochs, batch size, optimizer settings, learning rate schedules, and checkpoint loading.
- <b>Validation</b>: Sets up the validation process, including batch size, validation intervals, and re-ranking options.
- <b>Test</b>: Configures the testing phase, including embedding normalization, similarity algorithms, and paths to test images or models.
- <b>Tracking</b>: Defines settings for object tracking, including YOLO configurations, filtering thresholds, and output paths for bounding boxes and videos.
- <b>Database</b>: Configures the MongoDB database connection and collections for storing vehicle, camera, trajectory, and bounding box data.
- <b>Metrics</b>: Handles evaluation metrics for tracking and ReID tasks, including MOTA metrics, IoU thresholds, and prediction file management.
- <b>Pipeline</b>: Controls the overall pipeline execution, including video paths, ROI masks, target search, and database unification processes.

If you wish to override some parameters, create a `config.yaml` file .

## Full Pipeline of Tracking + ReID
To run the full pipeline, which involves Object Detection + Tracking + ReID, run:
```python
python pipeline.py <config_file>.yml
```

The YAML file specified here, must be a configuration file that contains the following section:
- <b>camera_x</b>: (path, roi, info, gt)
Contains the paths to the video/frames, the ROI, the informations and _possibly_ the ground truth.
- <b>layout</b>: (layout path)
Probably the most important and MANDATORY field. It contains geometric informations between all pairs of cameras, with FPS, [scales and offsets](https://github.com/regob/vehicle_mtmc/issues/5), compatibility (_matrix for determining whether a car can be seen from a camera to another_), dtmin (_minimum time for a vehicle to transition between two pairs of cams_) and dtmax. There's no pipeline without this field.
You can find examples of configuration files in the `config` folder.

For AICity22, it would be:
```python
python pipeline.py configs/cameras_s02_cityflow.yml
```

Please change the `config.py` file to adjust the MTMC (YOLO, Detector, Tracking & Pipeline configs).

## ReID
### Training
To train the model, run the following command in your terminal:<br>
`python -m reid.main <config_file>.yml`<br>
> N.B. If you do not specify a _.yml_ file, the script will use the pre-defined `config.py` file to set up the Dataset, Model, and Training parameters.
> N.B.2 For example, for [RPTM Training](https://openaccess.thecvf.com/content/WACV2023/papers/Ghosh_Relation_Preserving_Triplet_Mining_for_Stabilising_the_Triplet_Loss_In_WACV_2023_paper.pdf), you can call the existing config file `config_rptm.yml`

### Testing
To evaluate a trained model, use the following command:<br>
`python -m reid.test <config_test_file>.yml`<br>

> N.B. If you do not specify a _.yml_ file, the script will use the pre-defined `config_test.yml` file to set up the Model and Testing parameters.

> N.B. This script will load a pre-trained model specified in the `config_test.yml` file and either:
>- <b>Compare two specific images:</b> If `run_reid_metrics` is set to `False` in the config file, the script will compute the similarity or distance between the two images specified in `PATH_IMG_1` and `PATH_IMG_2`. The similarity metric (e.g., Euclidean distance or cosine similarity) is determined by the `SIMILARITY_ALGORITHM` setting in the config file.
>- <b>Run re-identification metrics:</b> If `run_reid_metrics` is set to `True`, the script will evaluate the model on the validation set and compute re-identification metrics.
>- <b>Run color metrics:</b> If `run_color_metrics` is set to `True`, the script will evaluate the color classification model on the validation set and compute color-related metrics.

>Additionally, if `stack_images` is set to True and `run_reid_metrics` is `False`, the script will compute a similarity matrix for all images in the test directory and display it as a heatmap.
>For this reason, change the Test Configuration accordingly.
