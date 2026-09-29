"""Reuse completed experiments with matching model, training, and data settings."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .model import create_model
from .train import train_model


def _dataset_description(dataset):
    """Describe sample membership and transforms without reading every image."""
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        return {'type': 'subset', 'indices': list(dataset.indices),
                'source': _dataset_description(dataset.dataset)}
    if hasattr(dataset, 'datasets'):
        return {'type': 'concat',
                'sources': [_dataset_description(part) for part in dataset.datasets]}
    if hasattr(dataset, 'samples'):
        return {'type': 'image_folder', 'samples': dataset.samples,
                'transform': repr(dataset.transform)}
    raise TypeError(f'Unsupported dataset type for experiment cache: {type(dataset).__name__}')


def _history_from_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row['epoch'] = int(row['epoch'])
        for key, value in row.items():
            if key != 'epoch' and not key.endswith('_confusion_matrix'):
                row[key] = float(value)
    return rows


def load_or_train(name, *, num_classes, pretrained, mode, train_loader,
                  val_loader, criterion, device, epochs, lr, seed, output_dir,
                  force_train=False, legacy_checkpoint_path=None,
                  legacy_csv_path=None):
    """Return (best_model, history, reused).

    A completed run is reused only when its model, settings, dataset membership,
    and transforms match. This is not interrupted-training resume.
    """
    settings = {
        'name': name, 'num_classes': num_classes, 'pretrained': pretrained,
        'mode': mode, 'epochs': epochs, 'lr': lr, 'seed': seed,
        'optimizer': 'Adam', 'criterion': type(criterion).__name__,
        'train_batch_size': train_loader.batch_size,
        'val_batch_size': val_loader.batch_size,
        'train_data': _dataset_description(train_loader.dataset),
        'val_data': _dataset_description(val_loader.dataset),
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    run_dir = Path(output_dir) / f'{name}_{mode}_{fingerprint[:12]}'
    csv_path = run_dir / 'history.csv'
    checkpoint_path = run_dir / 'best_recall.pt'
    manifest_path = run_dir / 'manifest.json'
    model = create_model(name, num_classes=num_classes,
                         pretrained=pretrained, mode=mode).to(device)
    if (not force_train and legacy_checkpoint_path is not None and
            legacy_csv_path is not None and not manifest_path.exists()):
        old_checkpoint = Path(legacy_checkpoint_path)
        old_csv = Path(legacy_csv_path)
        if old_checkpoint.exists() and old_csv.exists():
            old_history = _history_from_csv(old_csv)
            saved = torch.load(old_checkpoint, map_location=device, weights_only=True)
            matching_sizes = (old_history and
                              sum(json.loads(old_history[-1]['train_confusion_matrix'])[0] +
                                  json.loads(old_history[-1]['train_confusion_matrix'])[1]) == len(train_loader.dataset) and
                              sum(json.loads(old_history[-1]['val_confusion_matrix'])[0] +
                                  json.loads(old_history[-1]['val_confusion_matrix'])[1]) == len(val_loader.dataset))
            if (len(old_history) == epochs and matching_sizes and
                    saved.get('learning_rates') == [lr] and
                    saved.get('epoch') == max(old_history, key=lambda row:
                        (row['val_recall'], row['val_f1'], -row['val_loss']))['epoch']):
                model.load_state_dict(saved['model_state_dict'])
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_checkpoint, checkpoint_path)
                shutil.copy2(old_csv, csv_path)
                manifest_path.write_text(json.dumps({'fingerprint': fingerprint,
                    'best_epoch': saved['epoch']}, indent=2), encoding='utf-8')
                print(f'Imported completed {name} experiment from previous checkpoint')
                return model, old_history, True
    if not force_train and all(p.exists() for p in (csv_path, checkpoint_path, manifest_path)):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('fingerprint') == fingerprint:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            history = _history_from_csv(csv_path)
            if len(history) == epochs and checkpoint['epoch'] == manifest['best_epoch']:
                print(f"Reused {name}: best epoch {checkpoint['epoch']}, "
                      f"val recall {checkpoint['val_recall']:.3f}, LR {lr}")
                return model, history, True
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                          device, epochs=epochs, csv_path=csv_path,
                          checkpoint_path=checkpoint_path)
    best = max(history, key=lambda row: (row['val_recall'], row['val_f1'],
                                         -row['val_loss']))
    manifest_path.write_text(json.dumps({'fingerprint': fingerprint,
                                         'best_epoch': best['epoch']}, indent=2),
                             encoding='utf-8')
    return model, history, False
