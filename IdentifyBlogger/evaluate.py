import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from IdentifyBlogger.neural.metrics import score, append_scores, avg_scores
from IdentifyBlogger.data.prepare_data import prepare_evaluation_data
from IdentifyBlogger.neural.train_loop import forward


def evaluate(model_path: str, data_path: str, preprocess_info_dir: str = None, batch_size: int = 64) -> None:
    """
    function to evaluate previously trained model
    :param model_path: path to model to evaluate
    :param data_path: path to data
    :param preprocess_info_dir: path to preprocessing information
    :param batch_size: batch size for data loader
    """
    data = pd.read_csv(data_path)
    test_dataset = prepare_evaluation_data(data, preprocess_info_dir)
    model = torch.load(model_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    with torch.no_grad():
        scores = None
        for encoded_text, lengths, labels in test_loader:
            y_pred = forward(model, encoded_text, lengths, labels.keys())

            batch_scores = score(y_pred, labels)
            if scores is None:
                scores = {}
                for label in batch_scores.keys():
                    scores[label] = {}
                    for metric in batch_scores[label].keys():
                        scores[label][metric] = []
                else:
                    scores = append_scores(scores, batch_scores,
                                           weight=len(lengths) / test_loader.batch_size)

    scores = avg_scores(scores)
    print(f"Test scores")
    for label in scores.keys():
        print(f"{label}: " + "".join([f"{metric}: {s:.2f} " for metric, s in scores[label].items()]))


if __name__ == '__main__':
    fire.Fire(evaluate)