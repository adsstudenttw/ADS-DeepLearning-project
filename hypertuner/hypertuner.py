import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import Dict
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import PaddedPreprocessor
from filelock import FileLock
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics, rnn_models
import torch

@dataclass
class HyperParameterConfig:
    input_size: float = 3
    output_size: int = 20
    tuning_directory: str
    data_directory: str
    hidden_size: int = tune.randint(16, 128)
    dropout: float = tune.uniform(0.0, 0.3)
    num_layers: int = tune.randint(2, 5)
    batchsize: int = 32

def train(config: Dict):
    data_directory = config["data_dir"]
    flowersdatasetfactory = DatasetFactoryProvider.create_factory(
        DatasetType.FLOWERS,
        data_directory=Path(config["data_dir"])
    )
    preprocesor = PaddedPreprocessor()

    with FileLock(data_directory / ".lock"):
        streamers = flowersdatasetfactory.create_datastreamer(
            batchsize=config["data_dir"],
            preprocesor=preprocesor,
        )
        train = streamers["train"]
        valid = streamers["valid"]
    
    accuracy = metrics.Accuracy
    model = rnn_models.GRUmodel(config)

    trainersettings = TrainerSettings(
        epochs=50,
        metrics=[accuracy],
        logdir=Path("."),
        train_steps=len(train),
        valid_steps=len(valid),
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        traindataloader=train.stream(),
        validdataloader=valid.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    trainer.loop()


if __name__ == "__main__":
    ray.init()

    data_directory = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_directory.exists():
        data_directory.mkdir(parents=True)
        logger.info(f"created {data_directory}")
    tuning_directory = Path("models/ray").resolve()

    configuration = HyperParameterConfig()

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    hyperband_bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    search_bohb = TuneBOHB()

    analysis = tune.run(
        train,
        config=configuration.__dict__,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=configuration.tuning_directory,
        num_samples=50,
        search_alg=search_bohb,
        scheduler=hyperband_bohb,
        verbose=1
    )