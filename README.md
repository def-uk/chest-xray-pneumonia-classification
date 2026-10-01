# Chest X-ray pneumonia classification

`notebooks/exploration.ipynb` contains EDA and the current small-CNN / pretrained-ResNet18 experiments. Reusable code lives in `src/data.py`, `src/model.py`, `src/train.py`, and `src/evaluate.py`. The four hand-written architectures are in `src/models/`.

Use `create_model('alexnet'|'vgg16'|'googlenet'|'resnet18', num_classes=2, pretrained=False)` for scratch models, or `pretrained=True, mode='frozen'|'partial'|'full'` for torchvision models with ImageNet weights. The hand-written models are for structure experiments; pretrained models use torchvision implementations for reliable weights.

Run the notebook from the project root or `notebooks/` directory. Install dependencies with `pip install -r requirements.txt`. KaggleHub downloads the dataset on first use. The notebook splits the original train+val data with seed 42 and evaluates the provided test set after validation-based selection.

The final one-file submission notebook must inline or bundle the needed `src/` functions; this development notebook imports project files.

## Reuse completed experiments

`src/experiment.py` provides `load_or_train(...)`. It caches the best validation-recall checkpoint, epoch CSV, and a settings/data fingerprint under `outputs/experiments/`. Re-running the same notebook settings loads the completed model and history instead of training again. Changing the model, fine-tuning mode, learning rate, epochs, seed, data split, or transforms creates a new run. `force_train=True` starts a fresh run. This is result reuse, not interrupted-training resume.

The train+original-val pool is now split by patient identifier with `StratifiedGroupKFold`. Same-patient images cannot cross train and validation. Previous image-level split checkpoints should not be reused; the notebook starts new experiments under a new data fingerprint.

## DenseNet and EfficientNet five-fold comparison

Notebook section 5 retains the previous top seven models, ranked by mean accuracy,
recall, and F1 descending, then recall/F1 standard deviations ascending. Section
5.2 adds torchvision DenseNet121 and EfficientNet-B0, each with full fine-tuning,
random initialization, and classifier-only training (30 fold runs total).
All six conditions use 10 epochs, batch size 32, seed 42, Adam LR 1e-4, and
ImageNet normalization. Patient folds and validation duplicate filtering match
the earlier experiments; the provided test set is not evaluated.

The same experiments can run independently of the notebook:

```sh
python -u -m src.cross_validation --dataset-root /path/to/dataset --output-dir outputs
```

`configs/baseline_top7_cv_summary.csv` preserves the supplied previous results
rounded to four decimals. Pass `--baseline outputs/baseline_top7_cv_summary.csv`
to use the full-precision summary after rerunning section 5.1.
Completed experiments reuse their checkpoints. Partial fold results are saved
to `outputs/densenet_efficientnet_cv_partial.csv`; final per-fold results and
the new-model summary are saved after all folds complete. The combined 13-row
table is written to `outputs/cv_comparison.csv`. Its mean metrics sort descending
and standard deviations ascending in the same priority order as the top-seven
selection.

Section 6 now compares only DenseNet121, EfficientNet-B0, VGG16, and ResNet18
with full fine-tuning. It loads the completed first-fold checkpoints with
`checkpoint_only=True` (no fallback training), then independently perturbs one
validation image per class with brightness, contrast, and rotation. The selected
comparison, prediction details, and sensitivity summaries are saved as
`outputs/selected_cv_comparison.csv`, `outputs/selected_augmentation_details.csv`,
and `outputs/selected_augmentation_summary.csv`. These two-image results help
form augmentation hypotheses; they are not full-validation robustness metrics.
