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
        self.lstm = nn.LSTM(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.hidden_dim*2, config.output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.config.num_layers*2, x.size(0), self.config.hidden_dim).to(self.config.device) # hidden state
        c0 = torch.zeros(self.config.num_layers*2, x.size(0), self.config.hidden_dim).to(self.config.device) # cell state
        
        x = x[:, None, :]
        #print ("init x shape: ", x.shape)
        out, _ = self.lstm(x, (h0, c0))
        #print ("lstm out shape: ", out.shape)
        out = self.fc(out[:, -1, :])
        #print ("linear out shape: ", out.shape)
        out = self.relu(out)
        return out