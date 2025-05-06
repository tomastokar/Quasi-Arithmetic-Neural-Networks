import torch
import torch.nn as nn

class MyCrossEntropyLoss(nn.Module):
    """
        Wrapper for copmputing CrossEntropy loss using (batch, 1)-shaped targets
    """
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits, y):
        return self.loss_fn(logits, y.flatten())


class MultiLabelAccuracy(nn.Module):
    def __init__(self, threshold=0.5, balanced=False):
        """
        Computes per-class accuracy and averages it. Optionally, computes per-class balanced accuracy.

        Args:
            threshold (float): Threshold to convert probabilities to binary (default: 0.5).
            balanced (bool): Whether to compute per-class balanced accuracy (default: False).
        """
        super(MultiLabelAccuracy, self).__init__()
        self.threshold = threshold
        self.balanced = balanced

    def forward(self, logits, targets):
        """
        Computes average per-class accuracy or balanced accuracy.

        Args:
            logits (torch.Tensor): Raw model outputs (shape: batch_size x num_labels).
            targets (torch.Tensor): Ground-truth labels (binary) (shape: batch_size x num_labels).

        Returns:
            avg_accuracy (float): Average per-class accuracy or balanced accuracy.
            per_class_accuracy (torch.Tensor): Accuracy for each class (shape: num_labels).
        """
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        predictions = (probabilities > self.threshold).float()  # Convert to binary predictions

        if self.balanced:
            # Compute per-class sensitivity (recall for positives) and specificity (recall for negatives)
            tp = ((predictions == 1) & (targets == 1)).float().sum(dim=0)  # True Positives
            fn = ((predictions == 0) & (targets == 1)).float().sum(dim=0)  # False Negatives
            tn = ((predictions == 0) & (targets == 0)).float().sum(dim=0)  # True Negatives
            fp = ((predictions == 1) & (targets == 0)).float().sum(dim=0)  # False Positives

            # Sensitivity (recall for positive class)
            sensitivity = tp / (tp + fn + 1e-8)  # Avoid division by zero

            # Specificity (recall for negative class)
            specificity = tn / (tn + fp + 1e-8)  # Avoid division by zero

            # Balanced accuracy per class
            per_class_accuracy = (sensitivity + specificity) / 2
        else:
            # Standard per-class accuracy
            correct_per_class = (predictions == targets).float().sum(dim=0)
            total_per_class = targets.shape[0]
            per_class_accuracy = correct_per_class / total_per_class

        avg_accuracy = per_class_accuracy.mean()

        return avg_accuracy


def multi_class_accuracy(logits, targets):
    """
    Computes classification accuracy.

    Args:
        logits (torch.Tensor): Model predictions (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels (batch_size,).

    Returns:
        float: Accuracy (0.0 to 1.0).
    """
    # Get predicted class (index of max logit)
    preds = torch.argmax(logits, dim=1)

    # Compute accuracy
    correct = (preds == targets).sum()
    total = targets.size(0)

    return correct / total  # Accuracy as a float