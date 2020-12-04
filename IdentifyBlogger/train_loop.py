import time
from typing import Tuple, List, Union, Dict, Callable

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from IdentifyBlogger.BloggerDataset import collate
from IdentifyBlogger.metrics import score, append_scores, avg_scores


def forward(model: nn.Module, encoded_text: torch.Tensor, lengths: torch.Tensor, label_names: List[str]) \
        -> Dict[str, torch.Tensor]:
    """

    :param model:
    :param encoded_text:
    :param lengths:
    :param label_names:
    :return:
    """
    encoded_text, lengths = encoded_text.to(model.device), lengths.to(model.device)
    y_pred = model(encoded_text, lengths)
    return {name: result if name not in ["gender", "age"] else result.view(-1) for name, result in zip(label_names,
                                                                                                       y_pred)}


def compute_loss(y_pred: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], criterions_fncs: List[nn.Module]) \
        -> torch.Tensor:
    """

    :param y_pred:
    :param labels:
    :param criterions_fncs:
    :return:
    """
    labels = {k: v.to(y_pred[k].device) for k, v in labels.items()}
    losses = [fn(y_pred[name], labels[name]) for name, fn in zip(y_pred.keys(), criterions_fncs)]
    loss = None
    for l in losses:
        loss = l if loss is None else loss + l
    return loss


def backward(loss: torch.Tensor, *optimizers) -> None:
    """

    :param loss:
    :param optimizer:
    :return:
    """
    loss.backward()
    for o in optimizers:
        o.step()


def train_loop(model: nn.Module, train_dataset: Dataset, validation_dataset: Dataset = None, lr: float = 1e-3,
               batch_size: int = 32, n_epochs: int = 1, criterions: List[str] = None, num_workers: int = 0,
               score_every: int = 2, model_path: str = None, device: Union[str, torch.device] = 'cpu') -> None:
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
    optimizer_sparse = optim.SparseAdam(model.embedding.parameters(), lr=lr)
    optimizer_dense = optim.Adam(list(model.output_layers.parameters()) + list(model.lstm.parameters()), lr=lr)

    best_loss = 999.9

    print("Starting training!")
    for i in range(n_epochs):
        # train_losses = []
        start = time.time()
        for encoded_text, lengths, labels in tqdm(train_loader):
            optimizer_dense.zero_grad()
            optimizer_sparse.zero_grad()
            y_pred = forward(model, encoded_text, lengths, labels.keys())
            loss = compute_loss(y_pred, labels, criterion_fncs)
            # train_losses.append(loss.item())
            backward(loss, optimizer_dense, optimizer_sparse)

        # print(f"epoch: {i} train loss: {sum(train_losses) / len(train_losses):.4f} time: {time.time() - start:.2f}")

        with torch.no_grad():
            test_losses = []
            scores = None
            for encoded_text, lengths, labels in validation_loader:
                y_pred = forward(model, encoded_text, lengths, labels.keys())
                loss = compute_loss(y_pred, labels, criterion_fncs)
                test_losses.append(loss.item())

                if score_every:
                    if i % score_every == 0:
                        batch_scores = score(y_pred, labels)
                        if scores is None:
                            scores = {}
                            for label in batch_scores.keys():
                                scores[label] = {}
                                for metric in batch_scores[label].keys():
                                    scores[label][metric] = []
                            else:
                                scores = append_scores(scores, batch_scores,
                                                       weight=len(lengths) / validation_loader.batch_size)

            avg_test_loss = sum(test_losses) / len(test_losses)
            if model_path is not None and avg_test_loss < best_loss:
                torch.save(model, model_path)

            if score_every:
                if i % score_every == 0:
                    scores = avg_scores(scores)
                    print(f"epoch: {i} Validation scores")
                    for label in scores.keys():
                        print(f"{label}: " + "".join([f"{metric}: {s:.2f} " for metric, s in scores[label].items()]))

            # print(f"epoch: {i} test loss: {sum(test_losses) / len(test_losses):.4f}")
