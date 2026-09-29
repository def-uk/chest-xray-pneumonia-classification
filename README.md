# Chest X-ray pneumonia classification

`notebooks/exploration.ipynb` contains EDA and the current small-CNN / pretrained-ResNet18 experiments. Reusable code lives in `src/data.py`, `src/model.py`, `src/train.py`, and `src/evaluate.py`. The four hand-written architectures are in `src/models/`.

Use `create_model('alexnet'|'vgg16'|'googlenet'|'resnet18', num_classes=2, pretrained=False)` for scratch models, or `pretrained=True, mode='frozen'|'partial'|'full'` for torchvision models with ImageNet weights. The hand-written models are for structure experiments; pretrained models use torchvision implementations for reliable weights.

Run the notebook from the project root or `notebooks/` directory. Install dependencies with `pip install -r requirements.txt`. KaggleHub downloads the dataset on first use. The notebook splits the original train+val data with seed 42 and evaluates the provided test set after validation-based selection.

The final one-file submission notebook must inline or bundle the needed `src/` functions; this development notebook imports project files.

## Reuse completed experiments

`src/experiment.py` provides `load_or_train(...)`. It caches the best validation-recall checkpoint, epoch CSV, and a settings/data fingerprint under `outputs/experiments/`. Re-running the same notebook settings loads the completed model and history instead of training again. Changing the model, fine-tuning mode, learning rate, epochs, seed, data split, or transforms creates a new run. `force_train=True` starts a fresh run. This is result reuse, not interrupted-training resume.

The train+original-val pool is now split by patient identifier with `StratifiedGroupKFold`. Same-patient images cannot cross train and validation. Previous image-level split checkpoints should not be reused; the notebook starts new experiments under a new data fingerprint.
