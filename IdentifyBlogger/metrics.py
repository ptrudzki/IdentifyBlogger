import torch


def recall_macro(labels: torch.Tensor, pred_labels: torch.Tensor, n_labels: int) -> float:
    """

    :param labels:
    :param pred_labels:
    :param n_labels:
    :return:
    """
    recall = 0
    for i in range(n_labels):
        label = (labels == i).float()
        pred_label = (pred_labels == i).float()
        if label.sum() > 0 and pred_label.sum() > 0:
            true_positive = label * pred_label
            recall += true_positive.sum().item() / label.sum().item()

    return recall / n_labels


def f1_score_macro(labels: torch.Tensor, pred_labels: torch.Tensor, n_labels: int) -> float:
    """
    computes f1 score (with macro averaging)
    :param labels: ground truth labels
    :param pred_labels: predicted labels
    :param n_labels: number of unique possible labels
    :return: macro averaged f1 score
    """
    f1 = 0
    for i in range(n_labels):
        label = (labels == i).float()
        pred_label = (pred_labels == i).float()
        if label.sum() > 0 and pred_label.sum() > 0:
            true_positive = label * pred_label
            precision = true_positive.sum().item() / pred_label.sum().item()
            recall = true_positive.sum().item() / label.sum().item()
            if precision > 0 and recall > 0:
                f1 += 2 * precision * recall / (precision + recall)

    return f1 / n_labels


def accuracy(labels: torch.Tensor, pred_labels: torch.Tensor) -> float:
    """
    computes accuracy
    :param labels: ground truth labels
    :param pred_labels: predicted labels
    :return: accuracy
    """
    return (pred_labels == labels).sum().item() / labels.shape[0]