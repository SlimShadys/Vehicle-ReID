# Vehicle Re-Identification

## Configuration
The project starts by properly filling the `config.json` file:
- `data_path`: The data path of the Dataset (e.g. `"data"`)
- `dataset_name`: The Dataset name (e.g. `"veri-776"`, `"veri-wild"`, `"vehicle-id"`)
- `dataset_size`: The Dataset Test size (e.g. `"small"`, `"medium"`, `"large"`). _N.B. This only works for `"veri-wild"` and `"vehicle-id"`, as `"veri-776"` has a fixed test size._
- `model`: The model to use (e.g. `"resnet50`, `resnet152`, etc.)
- `pretrained`: Whether to load the pretrained version of the model (`True` / `False`)
- `device`: Whether to use a CUDA device or not (`cuda` or `cpu`)
- `epochs`: For how many epochs to train the Network (Default: `10`)
- `batch_size`: Batch size for the Network (Default: `32`)
- `learning_rate`: Learning rate for the Optimizer (Default: `1e-3`)

## Training

## Testing