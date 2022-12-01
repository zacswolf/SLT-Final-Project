import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler, dotdict
import pytorch_lightning as pl


class Dataset_Custom(Dataset):
    def __init__(self, config, flag="train", scaler=None):

        assert config.seq_len is not None

        config.pred_len = config.pred_len or config.seq_len  # Default to config.seq_len
        assert config.pred_len is not None
        assert config.pred_len > 0 and config.pred_len <= config.seq_len
        assert flag in ["train", "test", "val"]
        assert config.data_id is not None

        self.config = config
        self.flag = flag

        self.scaler = scaler or StandardScaler()

        self.__read_data__()

    def __read_data__(self):
        # TODO: Impliment scaling
        self.data_x = np.float32(
            np.load(
                f"data/dataset_{self.config.data_id}/{self.flag}_feats.npy",
                allow_pickle=True,
            )
        )  # T_flag x num_mf x num_windows
        self.data_y = np.float32(
            np.load(
                f"data/dataset_{self.config.data_id}/{self.flag}_labels.npy",
                allow_pickle=True,
            )
        )  # T_flag x 9

    def __getitem__(self, index):
        # TODO: Add some smart windowing
        # s_begin = index
        # s_begin = index * self.config.seq_len
        s_begin = index * self.config.pred_len
        s_end = s_begin + self.config.seq_len

        # r_begin = s_end - self.config.label_len
        # r_end = r_begin + self.config.label_len + self.config.pred_len

        seq_x = self.data_x[s_begin:s_end]

        seq_y = self.data_y[s_begin:s_end]

        return seq_x, seq_y

    def __len__(self):
        # TODO: Add some smart windowing
        # return len(self.data_x) - self.config.seq_len + 1
        # return len(self.data_x)//self.config.seq_len
        return (len(self.data_x) - self.config.seq_len) // self.config.pred_len + 1
        # + - - - + 0 0 0
        # 0 + - - - + 0 0
        # 0 0 + - - - + 0
        # 0 0 0 + - - - +
        # 8 - 5 + 1
        # + - - - + 0 0 0
        # 0 0 0 + - - - +
        # (8-5)//3 + 1 = 2
        # + - - - + 0 0 0 0
        # 0 0 0 + - - - + 0
        # (9-5)//3 + 1 = 2
        # + - - - + 0 0 0 0 0 0
        # 0 0 0 + - - - + 0 0 0
        # 0 0 0 0 0 0 + - - - +
        # (11-5)//3 + 1 = 3
        # + - - - +
        # (5-5)//3 + 1 = 3


def data_provider(config, flag):
    data_set = Dataset_Custom(config, flag=flag)

    if flag == "test" or flag == "val":
        shuffle_flag = False
        drop_last = True
        batch_size = config.batch_size
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = config.batch_size

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader


class DataModule(pl.LightningDataModule):
    def __init__(self, config: dotdict, num_workers: int = 0):
        super().__init__()

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self) -> int:
        return 9

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: str | None = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""
        # load datasets only if they're not loaded already

        self.data_train = Dataset_Custom(self.config, flag="train")
        self.data_val = Dataset_Custom(self.config, flag="val")
        self.data_test = Dataset_Custom(self.config, flag="test")

    def train_dataloader(self):
        print("train", len(self.data_train))
        return DataLoader(
            self.data_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        print("val", len(self.data_val))
        return DataLoader(
            self.data_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        print("test", len(self.data_test))
        return DataLoader(
            self.data_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
