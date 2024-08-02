# Vehicle Re-Identification

## Install proper libraries
In order to install the proper libraries, a `requirements.txt` file, is present.<br>
Simply run `pip install -r requirements.txt`.

## Configuration
The project has a dedicated `config.yml` file, including various configurations for:

- <b>Dataset</b>: Specifies the data path, dataset name, and size.
- <b>Model</b>: Defines the model architecture, pretrained status, device and various optimizations.
- <b>Loss function</b>: Sets parameters for the loss calculation, including alpha, margin, MALW and label smoothing.
- <b>Training</b>: Contains settings for the main training loop, such as number of epochs, batch size, learning rate and scheduler details. Also, contains settings for loading a checkpoint.
- <b>Validation</b>: Specifies batch size and validation interval for the validation process.
- <b>Test</b>: Includes settings for the testing phase, such as whether to normalize embeddings and the path to the model for validation.

## Training
To train the model, run the following command in your terminal:<br>
`python main.py config.yml`<br>
> N.B. If you do not specify a _.yml_ file, the script will use the pre-defined `config.yml` file to set up the Dataset, Model, and Training parameters.

## Testing
To evaluate a trained model, use the following command:<br>
`python test.py config.yml`<br>

> N.B. If you do not specify a _.yml_ file, the script will use the pre-defined `config.yml` file to set up the Model and Testing parameters.

> N.B. This script will load a pre-trained model specified in the `config.yml` file and either:
>- Compare two specific images if `run_reid_metrics` is set to `False` in the config file.
>- Run full re-identification metrics on the validation set if `run_reid_metrics` is set to `True`.