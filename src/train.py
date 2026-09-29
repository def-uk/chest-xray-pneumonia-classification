"""Training loop with per-epoch metrics and CSV logging."""
import csv
import json
from pathlib import Path
import torch

from .evaluate import classification_metrics, evaluate


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                epochs=10, csv_path=None, checkpoint_path=None):
    """Return epoch rows and restore the best validation-recall weights.

    Recall ties prefer higher F1, then lower validation loss. CSV confusion
    matrices use [[TN, FP], [FN, TP]].
    """
    history = []
    best_score = None
    best_epoch = None
    best_state = None
    output_path = Path(csv_path) if csv_path is not None else None
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        actual, predicted = [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            loss_sum += loss.item() * n
            count += n
            actual.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.detach().argmax(1).cpu().tolist())
        if count == 0:
            raise ValueError('Empty training loader')
        training = {'loss': loss_sum / count}
        training.update(classification_metrics(actual, predicted))
        validation = evaluate(model, val_loader, criterion, device)
        row = {'epoch': epoch}
        for prefix, metrics in (('train', training), ('val', validation)):
            for key in ('loss', 'accuracy', 'precision', 'recall', 'f1'):
                row[f'{prefix}_{key}'] = float(metrics[key])
            matrix = metrics['confusion_matrix'].tolist()
            row[f'{prefix}_confusion_matrix'] = json.dumps(matrix)
            row[f'{prefix}_tn'], row[f'{prefix}_fp'] = matrix[0]
            row[f'{prefix}_fn'], row[f'{prefix}_tp'] = matrix[1]
        history.append(row)
        score = (row['val_recall'], row['val_f1'], -row['val_loss'])
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            if checkpoint is not None:
                torch.save({
                    'epoch': epoch,
                    'learning_rates': [group['lr'] for group in optimizer.param_groups],
                    'val_recall': row['val_recall'],
                    'val_f1': row['val_f1'],
                    'model_state_dict': best_state,
                }, checkpoint)
        if output_path is not None:
            with output_path.open('w' if epoch == 1 else 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {row['train_loss']:.4f}, acc {row['train_accuracy']:.3f}, "
            f"P/R/F1 {row['train_precision']:.3f}/{row['train_recall']:.3f}/{row['train_f1']:.3f} | "
            f"val loss {row['val_loss']:.4f}, acc {row['val_accuracy']:.3f}, "
            f"P/R/F1 {row['val_precision']:.3f}/{row['val_recall']:.3f}/{row['val_f1']:.3f}"
        )
    model.load_state_dict(best_state)
    learning_rates = [group['lr'] for group in optimizer.param_groups]
    print(f"Best epoch {best_epoch}: val recall {best_score[0]:.3f}, "
          f"F1 {best_score[1]:.3f}, LR {learning_rates}")
    if checkpoint is not None:
        print(f"Checkpoint: {checkpoint.resolve()}")
    return history
