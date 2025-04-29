import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def calc_metrics(all_preds: np.ndarray, all_labels: np.ndarray, logger) -> tuple:
    """
    Calculate the F1 score, accuracy, and confusion matrix for the model predictions.
    """
    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info(f"Accuracy: {acc:.3f}")
    logger.info("Confusion Matrix:")
    print(cm)

    return f1, acc, cm


def evaluate_model(model: pl.LightningModule, test_loader: torch.utils.data.DataLoader, logger) -> tuple:
    """
    Evaluate the model on the test set and calculate metrics.
    """
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            preds = torch.argmax(y_hat, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1, acc, cm = calc_metrics(all_preds.numpy(), all_labels.numpy(), logger)
    return f1, acc, cm
