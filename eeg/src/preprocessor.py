import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Protocol, List

class PreprocessorProtocol(Protocol):
    def __call__(self, batch: List[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        ...

class BasePreprocessor(PreprocessorProtocol):
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        return torch.stack(X), torch.stack(y)


class PaddedPreprocessor(PreprocessorProtocol):
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X_ = pad_sequence(X, batch_first=True, padding_value=0)
        return X_, torch.tensor(y)

class WindowingPreprocessor(PreprocessorProtocol):
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)

        X_windowed = [self.window_sequence(seq) for seq in X]
        X_padded = pad_sequence(X_windowed, batch_first=True, padding_value=0)

        return X_padded, torch.stack(y)

    def window_sequence(self, sequence):
        windows = [sequence[i:i + self.window_size] for i in range(0, len(sequence), self.window_size)]
        return torch.stack(windows)