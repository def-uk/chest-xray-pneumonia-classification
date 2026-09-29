"""Classification metrics for training, validation, and final evaluation."""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)


def classification_metrics(actual, predicted, positive_label=1):
    """Binary metrics; by default class index 1 is PNEUMONIA."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average='binary', pos_label=positive_label,
        zero_division=0,
    )
    return {
        'accuracy': accuracy_score(actual, predicted),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(actual, predicted, labels=[0, 1]),
    }


def evaluate(model, loader, criterion, device, class_names=None):
    model.eval()
    total_loss, count = 0.0, 0
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            count += labels.size(0)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    if count == 0:
        raise ValueError('Empty evaluation loader')
    result = {'loss': total_loss / count}
    result.update(classification_metrics(actual, predicted))
    if class_names is not None:
        result['report'] = classification_report(
            actual, predicted, labels=[0, 1], target_names=class_names,
            output_dict=True, zero_division=0,
        )
    return result


def plot_confusion_matrix(matrix, class_names, title='Confusion matrix'):
    """Plot actual classes on rows and predicted classes on columns."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel='Predicted', ylabel='Actual', title=title)
    fig.tight_layout()
    return fig, ax


def plot_loss_history(history, title='Training and validation loss'):
    """Plot epoch losses and mark the checkpoint selected by validation recall."""
    if not history:
        raise ValueError('Training history is empty')
    best = max(history, key=lambda row: (row['val_recall'], row['val_f1'],
                                         -row['val_loss']))
    epochs = [row['epoch'] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [row['train_loss'] for row in history], marker='o', label='Train loss')
    ax.plot(epochs, [row['val_loss'] for row in history], marker='o', label='Validation loss')
    ax.axvline(best['epoch'], color='gray', linestyle='--', alpha=0.8,
               label=f"Best recall: epoch {best['epoch']}")
    ax.set(xlabel='Epoch', ylabel='Loss', title=title)
    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()
    return fig, ax
