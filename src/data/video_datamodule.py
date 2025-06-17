from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .components.train import create_train_dataset
from .components.test import create_test_dataset
import torch
from hydra.utils import get_class

class VideoDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
        pin_memory,
        # Common args
        base_model_name,
        fps,
        use_start_end,
        # TrainDataset
        train_class,
        train_split,
        train_pattern,
        train_step,
        # ValDataset
        val_class,
        val_split,
        val_pattern,
        val_step,
        # TestDataset
        test_class,
        test_split,
        test_refresh_interval=0,
        test_is_sequential=False,
        # Sampling
        train_sample_rate=1.,
        test_sample_rate=1.,
        # Common kwargs
        return_compressed=False,
        dataset_str='nextqa',
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        if stage == "fit" or stage is None:
            train_class = get_class(self.hparams.train_class)
            self.data_train = create_train_dataset(train_class)(
                pattern=self.hparams.train_pattern,
                split=self.hparams.train_split,
                base_model_name=self.hparams.base_model_name,
                fps=self.hparams.fps,
                step=self.hparams.train_step,
                use_start_end=self.hparams.use_start_end,
                return_compressed=self.hparams.return_compressed,
            )
            if self.hparams.train_sample_rate != 1.:
                steps = int(1 / self.hparams.train_sample_rate)
                self.data_train = torch.utils.data.Subset(
                    self.data_train,
                    indices=range(0, len(self.data_train), steps)
                )

            
            val_class = get_class(self.hparams.val_class)
            self.data_val = create_train_dataset(val_class)(
                pattern=self.hparams.val_pattern,
                split=self.hparams.val_split,
                base_model_name=self.hparams.base_model_name,
                fps=self.hparams.fps,
                step=self.hparams.val_step,
                use_start_end=self.hparams.use_start_end,
                return_compressed=self.hparams.return_compressed,
            )
            if self.hparams.test_sample_rate != 1.:
                steps = int(1 / self.hparams.test_sample_rate)
                self.data_val = torch.utils.data.Subset(
                    self.data_val,
                    indices=range(0, len(self.data_val), steps)
                )

        if stage == "test" or stage is None:
            test_class = get_class(self.hparams.test_class)
            self.data_test = create_test_dataset(test_class)(
                split=self.hparams.test_split,
                base_model_name=self.hparams.base_model_name,
                fps=self.hparams.fps,
                refresh_interval=self.hparams.test_refresh_interval,
                is_sequential=self.hparams.test_is_sequential,
                return_compressed=self.hparams.return_compressed,
                return_output_states=True,
            )

    def prepare_data(self) -> None:
        # TODO: maybe we can automate preprocessing steps from src.scripts
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )