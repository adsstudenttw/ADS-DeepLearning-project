from ast import Tuple
from typing import Protocol, Iterator, Optional, Callable, Sequence
import numpy as np
from eeg.src.eeg_dataset import DatasetProtocol

class DataStreamerProtocol(Protocol):
    def stream(self) -> Iterator:
        ...

class BaseDatastreamer(DataStreamerProtocol):
    def __init__(
        self, 
        dataset: DatasetProtocol, 
        batch_size: int, 
        preprocessor: Optional[Callable] = None
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

        if preprocessor == None:
            self.preprocessor = lambda x: zip(*x)
        else:
            self.preprocessor = preprocessor
        
        self.size = len(self.dataset)
        self.reset_index()
    
    def __len__(self) -> int:
        return int(len(self.dataset) / self.batch_size)
    
    def reset_index(self) -> None:
        self.index_list = np.random.permutation(self.size)
        self.index = 0
    
    def batchloop(self) -> Sequence[Tuple]:
        batch = []
        for _ in range(self.batch_size):
            x, y = self.dataset[self.index_list[self.index]]
            batch.append((x, y))
            self.index += 1
        return batch
    
    def stream(self) -> Iterator:
        if self.index > (self.size - self.batch_size):
            self.reset_index()
        batch = self.batchloop()
        X, Y = self.preprocessor(batch)
        yield X, Y

