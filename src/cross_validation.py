"""Five-fold DenseNet/EfficientNet experiments and the retained comparison."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import make_loaders
from .experiment import load_or_train

BASELINE_LABELS = [
    'VGG16 / Full fine-tuning', 'GoogLeNet / Full fine-tuning',
    'AlexNet / Full fine-tuning', 'AlexNet / Random initialization',
    'VGG16 / Random initialization', 'ResNet18 / Full fine-tuning', 'Custom CNN',
]
SORT_COLUMNS = ['mean_accuracy', 'mean_recall', 'mean_f1', 'std_recall', 'std_f1']
SORT_ASCENDING = [False, False, False, True, True]
SELECTED_MODEL_SPECS = {
    'DenseNet121 / Full fine-tuning': ('densenet121', True, 'full', True),
    'EfficientNet-B0 / Full fine-tuning': ('efficientnet_b0', True, 'full', True),
    'VGG16 / Full fine-tuning': ('vgg16', True, 'full', True),
    'ResNet18 / Full fine-tuning': ('resnet18', True, 'full', True),
}
NEW_MODEL_SPECS = {
    f'{label} / {condition}': (name, pretrained, mode, True)
    for label, name in [('DenseNet121', 'densenet121'), ('EfficientNet-B0', 'efficientnet_b0')]
    for condition, pretrained, mode in [
        ('Full fine-tuning', True, 'full'),
        ('Random initialization', False, 'full'),
        ('Classifier only', True, 'frozen'),
    ]
}


def summarize(results, folds=5):
    if results.duplicated(['model', 'fold']).any():
        raise ValueError('Duplicate model/fold results')
    summary = results.groupby('model', sort=False).agg(
        folds=('fold', 'nunique'), mean_accuracy=('accuracy', 'mean'),
        mean_precision=('precision', 'mean'), mean_recall=('recall', 'mean'),
        std_recall=('recall', 'std'), mean_f1=('f1', 'mean'), std_f1=('f1', 'std'),
    )
    if not (summary['folds'] == folds).all():
        raise ValueError(f'Every model must have {folds} completed folds')
    return summary.sort_values(SORT_COLUMNS, ascending=SORT_ASCENDING)


def compare(baseline, new_summary):
    retained = baseline.loc[BASELINE_LABELS]
    combined = pd.concat([retained, new_summary])
    if combined.index.duplicated().any() or not (combined['folds'] == 5).all():
        raise ValueError('Comparison requires distinct models with five folds each')
    return combined.sort_values(SORT_COLUMNS, ascending=SORT_ASCENDING)


def run_new_models(dataset_root, output_dir, *, epochs=10, batch_size=32,
                   seed=42, lr=1e-4, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu'))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device}; 6 conditions x 5 folds x {epochs} epochs', flush=True)
    base = make_loaders(dataset_root, batch_size=batch_size, seed=seed, pretrained=True)
    sources = base['train'].dataset.dataset.datasets
    paths = [path for source in sources for path, _ in source.samples]
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with open(path, 'rb') as image:
            for chunk in iter(lambda: image.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[path] = digest.hexdigest()
    del base
    rows = []
    seen = set()
    for fold in range(5):
        data = make_loaders(dataset_root, batch_size=batch_size, seed=seed,
                            val_fraction=0.2, pretrained=True, fold_index=fold)
        train_indices = data['train'].dataset.indices
        val_indices = data['val'].dataset.indices
        if seen.intersection(val_indices):
            raise RuntimeError('Validation fold overlap')
        seen.update(val_indices)
        train_hashes = {hashes[paths[i]] for i in train_indices}
        kept = [i for i in val_indices if hashes[paths[i]] not in train_hashes]
        if not kept:
            raise ValueError('Empty validation fold after duplicate removal')
        data['val'] = DataLoader(Subset(data['val'].dataset.dataset, kept),
                                 batch_size=batch_size, shuffle=False)
        print(f'Fold {fold + 1}: validation={len(kept)}, duplicates removed={len(val_indices) - len(kept)}', flush=True)
        for label, (name, pretrained, mode, _) in NEW_MODEL_SPECS.items():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            train_loader = DataLoader(data['train'].dataset, batch_size=batch_size,
                                      shuffle=True, generator=torch.Generator().manual_seed(seed))
            print(f'[{fold + 1}/5] {label}', flush=True)
            model, history, reused = load_or_train(
                name, num_classes=len(data['classes']), pretrained=pretrained,
                mode=mode, architecture_source='torchvision', train_loader=train_loader,
                val_loader=data['val'], criterion=nn.CrossEntropyLoss(), device=device,
                epochs=epochs, lr=lr, seed=seed, output_dir=output_dir / 'experiments',
            )
            # The restored checkpoint is exactly this epoch; no second evaluation needed.
            best = max(history, key=lambda row: (row['val_recall'], row['val_f1'], -row['val_loss']))
            rows.append(dict(model=label, fold=fold + 1, val_size=len(kept), reused=reused,
                             **{metric: best[f'val_{metric}'] for metric in
                                ('accuracy', 'precision', 'recall', 'f1')}))
            pd.DataFrame(rows).to_csv(output_dir / 'densenet_efficientnet_cv_partial.csv', index=False)
            model.to('cpu')
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
    if len(seen) != len(paths):
        raise RuntimeError('Not every sample belongs to a validation fold')
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / 'densenet_efficientnet_cv_results.csv', index=False)
    summary = summarize(results)
    summary.to_csv(output_dir / 'densenet_efficientnet_cv_summary.csv')
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--baseline', default='configs/baseline_top7_cv_summary.csv')
    args = parser.parse_args()
    baseline = pd.read_csv(args.baseline, index_col='model')
    baseline.loc[BASELINE_LABELS]  # Validate before starting expensive training.
    _, summary = run_new_models(args.dataset_root, args.output_dir)
    comparison = compare(baseline, summary)
    comparison.to_csv(Path(args.output_dir) / 'cv_comparison.csv')
    print(comparison.round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()
