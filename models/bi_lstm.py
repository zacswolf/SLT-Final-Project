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
        self.conv1 = nn.Conv2d(in_channels=self.config.input_dim, out_channels=128, kernel_size=(2,2), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2,2), stride=(1,1))
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=None)
        self.drop1 = nn.Dropout(0.4)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2,2), stride=(1,1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2,2), stride=(1,1))
        self.maxPool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=None)
        self.drop2 = nn.Dropout(0.4)

        # self.conv5 = nn.Conv2d(out_channels=5, kernel_size=(2,2), stride=(1,1))
        # self.conv6 = nn.Conv2d(out_channels=5, kernel_size=(2,2), stride=(1,1))
        # self.maxPool3 = nn.MaxPool3d(kernel_size=(2,2), stride=1)
        # self.drop3 = nn.Dropout(0.4)

        #self.reshape = 

        self.lstm1 = nn.LSTM(input_size=2, hidden_size=self.config.hidden_dim, num_layers=self.config.num_layers, batch_first=True, bidirectional=True, dropout=0.4)
        self.linear = nn.LazyLinear(out_features=self.config.output_dim) # in features was 256 from LSTM
        
    def forward(self, x):
        x = torch.squeeze(x,-3)
        print ("input: ", x.shape)
        og_shape = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxPool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxPool2(x)
        x = self.drop2(x)

        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.maxPool3(x)
        # x = self.drop3(x)

        hidden_state = torch.randn(3*2, self.config.input_dim, self.config.hidden_dim)
        cell_state = torch.randn(3*2, self.config.input_dim, self.config.hidden_dim)
        hidden = (hidden_state, cell_state)

        #self.reshape = 
        x, _ = self.lstm1(x, hidden)
        out = self.linear(x)
        return out