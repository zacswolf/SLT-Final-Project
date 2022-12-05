# 2.2 bi-LSTM Model Architecture
import torch
import torch.nn as nn
from utils.tools import dotdict

class bi_LSTM(nn.Module):
    def __init__(self, config):
        super(bi_LSTM, self).__init__()
        
        # Add defaults
        defaults = {"num_layers": 2, "model_type": "biLSTM"}
        config = dotdict(defaults | config)
        
        # Check params
        assert config.input_dim is not None
        assert config.hidden_dim is not None
        assert config.output_dim is not None
        assert config.model_type == "biLSTM"

        self.config = config

        # Create layers
        self.conv1 = nn.Conv2d(in_channels=self.config.input_dim, out_channels=x, kernel_size=(2,2), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=self.config.input_dim, out_channels=x, kernel_size=(2,2), stride=(1,1))
        self.maxPool1 = nn.MaxPool3d(kernel_size=(2,2), stride=1)
        self.drop1 = nn.Dropout(0.4)

        self.conv3 = nn.Conv2d(in_channels=self.config.input_dim, out_channels=x, kernel_size=(2,2), stride=(1,1))
        self.conv4 = nn.Conv2d(in_channels=self.config.input_dim, out_channels=x, kernel_size=(2,2), stride=(1,1))
        self.maxPool2 = nn.MaxPool3d(kernel_size=(2,2), stride=1)
        self.drop2 = nn.Dropout(0.4)

        self.conv5 = nn.Conv2d(in_channels=self.config.input_dim, out_channels=x, kernel_size=(2,2), stride=(1,1))
        self.conv6 = nn.Conv2d(in_channels=self.config.input_dim, out_channels=x, kernel_size=(2,2), stride=(1,1))
        self.maxPool3 = nn.MaxPool3d(kernel_size=(2,2), stride=1)
        self.drop3 = nn.Dropout(0.4)

        #self.reshape = 

        self.lstm1 = nn.LSTM(input_size=x, hidden_size=self.config.hidden_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.4)

        self.linear = nn.Linear(in_features=x out_features=self.config.output_dim)
        
    def forward(self, x):
        
        print ("input: ", x.shape)

        return out