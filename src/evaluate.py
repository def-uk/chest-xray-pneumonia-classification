"""Evaluation metrics; no optimizer updates."""
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
    result = {'loss': total_loss / count, 'accuracy': accuracy_score(actual, predicted)}
    if class_names is not None:
        indices = list(range(len(class_names)))
        result['report'] = classification_report(actual, predicted, labels=indices,
                                                 target_names=class_names, output_dict=True,
                                                 zero_division=0)
        result['confusion_matrix'] = confusion_matrix(actual, predicted, labels=indices)
    return result
