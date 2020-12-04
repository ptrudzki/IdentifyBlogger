import torch
from typing import Dict

from torch import nn


def recall_macro(labels: torch.Tensor, pred_labels: torch.Tensor, n_labels: int) -> float:
    """
    computes recall score (with macro averaging)
    :param labels: ground truth labels
    :param pred_labels: predicted labels
    :param n_labels: number of unique possible labels
    :return: macro averaged recall score
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


def score(y_pred: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
    """
    scores every output with adequate metrics, accuracy, recall and f1 for categorical variables and rmse for
    continous age
    :param y_pred: network predictions for each output
    :param labels: labels for each output
    :return: scores for every output and every metric
    """
    y_pred = {k: v.detach().cpu() for k, v in y_pred.items()}
    labels = {k: v.detach().cpu() for k, v in labels.items()}
    scores = {l: {} for l in labels.keys()}
    categorical_metrics = {"accuracy": accuracy, "recall": recall_macro, "f1": f1_score_macro}
    regression_metrics = {"rmse": lambda y_p, y_l: torch.sqrt(nn.MSELoss()(y_p, y_l)).item()}
    for label in scores.keys():
        if label == "age":
            for name, metric in regression_metrics.items():
                scores[label][name] = metric(y_pred[label], labels[label])
        else:
            for name, metric in categorical_metrics.items():
                pred_shape = 1 if len(y_pred[label].shape) == 1 else y_pred[label].shape[1]
                kwargs = {} if name == "accuracy" else {"n_labels": pred_shape}
                if pred_shape > 1:
                    pred = y_pred[label].argmax(1)
                else:
                    pred = (y_pred[label] > 0.5).float()
                scores[label][name] = metric(pred, labels[label], **kwargs)

    return scores


def append_scores(scores: Dict[str, Dict], new_scores: Dict[str, Dict], weight: float = 1.0) -> Dict[str, Dict]:
    """
    appends list of batch scores
    :param scores: previous scores
    :param new_scores: scores to add
    :param weight: weigth of new scores
    :return: appended scores
    """
    for label in scores.keys():
        for metric in scores[label].keys():
            scores[label][metric].append(new_scores[label][metric] * weight)
    return scores


def avg_scores(scores: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    averages scores
    :param scores: scores to average
    :return: averaged scores
    """
    for label in scores.keys():
        for metric in scores[label].keys():
            scores[label][metric] = sum(scores[label][metric]) / len(scores[label][metric])
    return scores
