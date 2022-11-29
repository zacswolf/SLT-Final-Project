import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler, dotdict


class Dataset_Custom(Dataset):
    def __init__(self, config, flag='train', scaler=None):

        assert config.seq_len is not None

        config.pred_len = config.pred_len or config.seq_len # Default to config.seq_len
        assert config.pred_len is not None
        assert config.pred_len > 0 and config.pred_len <= config.seq_len
        assert flag in ['train', 'test', 'val']
        assert config.data_id is not None
    
        self.config = config
        self.flag = flag

        self.scaler = scaler or StandardScaler()

        self.__read_data__()

    def __read_data__(self):
        # TODO: Impliment scaling
        self.data_x = np.load(f"data/dataset_{self.config.data_id}/{self.flag}_feats.npy", allow_pickle=True) # T_flag x num_mf x num_windows
        self.data_y = np.load(f"data/dataset_{self.config.data_id}/{self.flag}_labels.npy", allow_pickle=True) # T_flag x 9
    
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
        return (len(self.data_x) - self.config.seq_len)//self.config.pred_len + 1
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




def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Dataset_Custom(args, flag=flag)

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader