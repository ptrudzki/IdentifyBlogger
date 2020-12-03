import time
from typing import Tuple, List, Union, Dict, Callable

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from IdentifyBlogger.BloggerDataset import collate
from IdentifyBlogger.metrics import accuracy, recall_macro, f1_score_macro


def _forward(model: nn.Module, encoded_text: torch.Tensor, lengths: torch.Tensor, label_names: List[str]) \
        -> Dict[str, torch.Tensor]:
    """

    :param model:
    :param encoded_text:
    :param lengths:
    :param label_names:
    :return:
    """
    # encoded_text, lengths = encoded_text.to(model.device), lengths.to(model.device)
    y_pred = model(encoded_text, lengths)
    return {name: result if name not in ["gender", "age"] else result.view(-1) for name, result in zip(label_names,
                                                                                                       y_pred)}


def _compute_loss(y_pred: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], criterions_fncs: List[nn.Module]) \
        -> torch.Tensor:
    """

    :param y_pred:
    :param labels:
    :param criterions_fncs:
    :return:
    """
    # labels = {k: v.to(y_pred[k].device) for k, v in labels.items()}
    # nn.BCELoss()(y_pred["gender"].view(-1), labels["gender"].float())
    losses = [fn(y_pred[name], labels[name]) for name, fn in zip(y_pred.keys(), criterions_fncs)]
    loss = None
    for l in losses:
        loss = l if loss is None else loss + l
    return loss


def _score(y_pred: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """

    :param y_pred:
    :param labels:
    :return:
    """
    labels = {k: v.to(y_pred[k].device) for k, v in labels.items()}
    scores = {"accuracy": [], "recall": [], "f1": []}
    metrics = {"accuracy": accuracy, "recall": recall_macro, "f1": f1_score_macro}
    for label in labels.keys():
        for metric_name, metric in metrics.items():
            scores[metric_name].append(metric(y_pred[label], label[label]))
    return {k: sum(v) / len(v) for k, v in scores.items()}


def _backward(loss: torch.Tensor, *optimizers) -> None:
    """

    :param loss:
    :param optimizer:
    :return:
    """
    loss.backward()
    # for i, l in enumerate(loss):
    #     if i != len(loss) - 1:
    #         l.backward(retain_graph=True)
    #     else:
    #         l.backward()
    for o in optimizers:
        o.step()


def train_loop(model: nn.Module, train_dataset: Dataset, validation_dataset: Dataset = None, lr: float = 1e-3,
               batch_size: int = 32, n_epochs: int = 1, criterions: List[str] = None, num_workers: int = 0,
               score_every: int = None, device: Union[str, torch.device] = 'cpu') -> None:
    """

    :param model:
    :param train_dataset:
    :param validation_dataset:
    :param lr:
    :param batch_size:
    :param n_epochs:
    :param criterions:
    :param num_workers:
    :param score_every:
    :param device:
    :return:
    """
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              collate_fn=collate, pin_memory=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                   collate_fn=collate, pin_memory=False)

    criterion_fncs = [getattr(nn, c)() for c in criterions]
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_sparse = optim.SparseAdam(model.embedding.parameters(), lr=lr)
    # optimizer_sparse = optim.SparseAdam(list(model.embedding.parameters()) + list(model.lstm.parameters()), lr=lr)
    # optimizer_dense = optim.Adam(model.output_layers.parameters(), lr=lr)
    optimizer_dense = optim.Adam(list(model.output_layers.parameters()) + list(model.lstm.parameters()), lr=lr)

    print("Starting training!")
    for i in range(n_epochs):
        train_losses = []
        start = time.time()
        for encoded_text, lengths, labels in tqdm(train_loader):
            # optimizer.zero_grad()
            optimizer_dense.zero_grad()
            optimizer_sparse.zero_grad()
            y_pred = _forward(model, encoded_text, lengths, labels.keys())
            loss = _compute_loss(y_pred, labels, criterion_fncs)
            # loss, losses = _compute_loss(y_pred, labels, criterion_fncs)
            # train_losses.append(loss.item())
            train_losses.append(loss)
            # _backward(loss, optimizer)
            _backward(loss, optimizer_dense, optimizer_sparse)
            # _backward(losses, optimizer_dense, optimizer_sparse)
            if score_every is not None:
                if i % score_every == 0:
                    scores = _score(y_pred, labels)
                    print(f'avg train scores: accuracy: {scores["accuracy"]:.4f}; recall: {scores["recall"]:.4f}; '
                          f'f1: {scores["f1"]:.4f}')

        print(f"epoch: {i} train loss: {sum(train_losses) / len(train_losses):.4f} time: {time.time() - start:.2f}")

        with torch.no_grad():
            test_losses = []
            for encoded_text, lengths, labels in validation_loader:
                y_pred = _forward(model, encoded_text, lengths, labels.keys())
                loss = _compute_loss(y_pred, labels, criterion_fncs)
                test_losses.append(loss.item())
                if score_every is not None:
                    if i % score_every == 0:
                        scores = _score(y_pred, labels)
                        print(f'avg test scores: accuracy: {scores["accuracy"]:.4f}; recall: {scores["recall"]:.4f}; '
                              f'f1: {scores["f1"]:.4f}')

            print(f"epoch: {i} test loss: {sum(test_losses) / len(test_losses):.4f}")
