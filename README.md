# Chest X-ray pneumonia classification

`notebooks/exploration.ipynb` contains EDA and the current small-CNN / pretrained-ResNet18 experiments. Reusable code lives in `src/data.py`, `src/model.py`, `src/train.py`, and `src/evaluate.py`. The four hand-written architectures are in `src/models/`.

Use `create_model('alexnet'|'vgg16'|'googlenet'|'resnet18', num_classes=2, pretrained=False)` for scratch models, or `pretrained=True, mode='frozen'|'partial'|'full'` for torchvision models with ImageNet weights. The hand-written models are for structure experiments; pretrained models use torchvision implementations for reliable weights.

Run the notebook from the project root or `notebooks/` directory. Install dependencies with `pip install -r requirements.txt`. KaggleHub downloads the dataset on first use. The notebook splits the original train+val data with seed 42 and evaluates the provided test set after validation-based selection.

The final one-file submission notebook must inline or bundle the needed `src/` functions; this development notebook imports project files.
