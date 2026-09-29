"""Shared training loop for all experiments."""
import torch
from .evaluate import evaluate


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            count += n
        validation = evaluate(model, val_loader, criterion, device)
        row = {'epoch': epoch, 'train_loss': loss_sum / count,
               'train_accuracy': correct / count, 'val_loss': validation['loss'],
               'val_accuracy': validation['accuracy']}
        history.append(row)
        print(f"Epoch {epoch}/{epochs}: train acc {row['train_accuracy']:.3f}, val acc {row['val_accuracy']:.3f}")
    return history
