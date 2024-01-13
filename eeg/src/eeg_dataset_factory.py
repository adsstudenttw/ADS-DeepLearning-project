from mads_datasets import datatools
from pathlib import Path
from scipy.io import arff
from typing import Mapping
from abc import ABC
from pydantic import BaseModel
from eeg.src.eeg_dataset import DatasetProtocol, EegDataset

class DatasetSettings(BaseModel):
    dataset_url: str
    data_dir: Path
    filename: Path
    name: str
    unzip: bool
    
eegDatasetSettings = DatasetSettings(
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff",
    data_dir = Path("../data/eeg").resolve(),
    filename = Path("eeg.arff"),
    name = "EEG",
    unzip = False
)

class AbstractDatasetFactory(ABC):
    def __init__(self, settings: DatasetSettings) -> None:
        self.settings = settings
    
    def download_data(self) -> None:
        data_dir = self.settings.data_dir
        filename = self.settings.filename
        data_path = data_dir / filename

        if not data_path.exists():
            data_dir.mkdir(parents=True)
            datatools.get_file(
                data_dir=data_dir, 
                filename=filename, 
                url=self.settings.dataset_url, 
                unzip=False
            )

class EegDatasetFactory(AbstractDatasetFactory):
    def __init__(self, settings: DatasetSettings = eegDatasetSettings) -> None:
        super().__init__(settings)
        self.datasets = Mapping[str, DatasetProtocol]
    
    def create_dataset(self) -> Mapping[str, DatasetProtocol]:
        self.download_data()

        data_path = self.settings.data_dir / self.settings.filename
        dataset = arff.loadarff(data_path)[0]

        split = int(0.8 * len(dataset))
        train_dataset = EegDataset(dataset[:split])
        valid_dataset = EegDataset(dataset[split:])

        datasets = {
            "train": train_dataset,
            "valid": valid_dataset
        }
        self.datasets = datasets

        return datasets