# Vehicle Re-Identification

## Install proper libraries
In order to install the proper libraries, please first install TensorFlow and PyTorch. The following guidelines are made for Windows 10/11. Adjust accordingly.

### TensorFlow

```bat
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
```

### PyTorch
```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Once everything has been installed, simply run the `requirements.txt` file with:
```bat
pip install -r requirements.txt
```

## Configuration
The project has a dedicated `config.py` file where all the default parameters have been set, such as:

- <b>Misc</b>: Specify the seed, output directory and whether to use AMP (Experimental)
- <b>Dataset</b>: Specifies the data path, dataset name, size and sampler.
- <b>Model</b>: Defines the model architecture, pretrained status, device and various optimizations.
- <b>Augmentation</b>: Specifies which Data Augmentation to apply for the Dataset
- <b>Loss</b>: Sets parameters for the loss calculation, including alpha, RPTM type, Triplet Margin, MALW and label smoothing.
- <b>Training</b>: Contains settings for the main training loop, such as number of epochs, batch size, learning rate and scheduler details. Also, contains settings for loading a checkpoint.
- <b>Validation</b>: Specifies batch size and validation interval for the validation process. Contains also Re-Ranking and whether to save images of the final rankings.
- <b>Test</b>: Includes settings for the testing phase, such as whether to normalize embeddings, run validation on a Dataset and the path to the model for validation.

If you wish to override some parameters, create a `config.yaml` file (or use an already existing one, such as `config_rptm.yml` for [RPTM Training](https://openaccess.thecvf.com/content/WACV2023/papers/Ghosh_Relation_Preserving_Triplet_Mining_for_Stabilising_the_Triplet_Loss_In_WACV_2023_paper.pdf)).

## Full Pipeline of Tracking + ReID
To run the full pipeline, which involves Object Detection + Tracking + ReID, run:
<br>
`python pipeline.py <config_file>.yml`<br>

## ReID
### Training
To train the model, run the following command in your terminal:<br>
`python -m reid.main <config_file>.yml`<br>
> N.B. If you do not specify a _.yml_ file, the script will use the pre-defined `config.py` file to set up the Dataset, Model, and Training parameters.

### Testing
To evaluate a trained model, use the following command:<br>
`python -m reid.test <config_test_file>.yml`<br>

> N.B. If you do not specify a _.yml_ file, the script will use the pre-defined `config_test.yml` file to set up the Model and Testing parameters.

> N.B. This script will load a pre-trained model specified in the `config_test.yml` file and either:
>- Compare two specific images defined in `path_img_1` / `path_img_2` if `run_reid_metrics` is set to `False` in the config file.
>- Run re-identification metrics on the validation set if `run_reid_metrics` is set to `True`.
>- Run color metrics on the validation set if `run_color_metrics` is set to `True`.
>
>For this reason, change the Test Configuration accordingly.