from typing import List

import torch
from torch import nn


class IdentityLSTM(nn.Module):
    """
    basic LSTM network with word encodings layer on input and fully connected layer with softmax activation on output
    """

    def __init__(self, vocab_size: int, output_dims: List[int] = None, embedding_size: int = 32, hidden_size: int = 512,
                 n_layers: int = 4, activations: List[str] = None) -> None:
        """
        Initializes IntentLSTM
        :param vocab_size: number of unique words in data
        :param output_dims: number of data classes
        :param embedding_size: size of embeddings per word in embedding layer
        :param hidden_size: size of hidden states in lstm layers
        :param n_layers: number of lstm layers
        """
        super(IdentityLSTM, self).__init__()
        if activations is None:
            activations = ["Sigmoid"]
        if output_dims is None:
            output_dims = [1]
        assert len(output_dims) == len(activations), f"length of output dims ({len(output_dims)}) must be equal to length of activations ({len(activations)})"
        self.embedding = nn.Embedding(vocab_size, embedding_size, sparse=True)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.output_layers = nn.ModuleList([self._get_output_layer(n_layers * hidden_size, d, a) for d, a in
                                            zip(output_dims, activations)])

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_outputs(self):
        return len(self.output_layers)

    @staticmethod
    def _get_output_layer(input_dim: int, output_dim: int, activation: str = None) -> nn.Sequential:
        """
        returns linear layer with specified activation
        :param input_dim:
        :param output_dim:
        :param activation:
        :return:
        """
        layers = [nn.Linear(input_dim, output_dim)]
        if activation is not None:
            act = getattr(nn, activation)
            kwargs = {} if activation != "Softmax" else {"dim": 1}
            layers.append(act(**kwargs))
        return nn.Sequential(*layers)

    def forward(self, encoded_text: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
        """
        forward pass
        :param encoded_text: padded pack of encoded phrases
        :param lengths: lengths of subsequent phrases in pack
        :return: network prediction
        """
        batch_size = lengths.shape[0]
        embedded = self.embedding(encoded_text)
        packed_embeded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        _, (hidden, cell) = self.lstm(packed_embeded)
        hidden = hidden.permute([1, 0, 2]).contiguous().view(batch_size, -1)
        outputs = [layer(hidden) for layer in self.output_layers]
        return outputs
