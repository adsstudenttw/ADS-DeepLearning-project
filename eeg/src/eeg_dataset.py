from ast import Tuple
import torch
from typing import Protocol, List, Any
from abc import abstractmethod
from numpy import ndarray

class DatasetProtocol(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Any:
        ...

class ProcessingDatasetProtocol(DatasetProtocol):
    def process_data(self) -> None:
        ...

class AbstractDataset(ProcessingDatasetProtocol):
    def __init__(self, data: Tuple) -> None:
        self.dataset: List = []
        self.process_data(data)
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple:
        return self.dataset[index]
    
    @abstractmethod
    def process_data(self, data: List) -> None:
        raise NotImplementedError

class EegDataset(AbstractDataset):
    def process_data(self, data: ndarray) -> None:

        for set in data:
            set = set.tolist()
            self.dataset.append((torch.tensor(set[:-1]), torch.tensor(int(set[-1]))))