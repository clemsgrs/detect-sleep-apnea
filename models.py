import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(p):

    model = None
    print(p.model)
    
    if p.model == 'rnn':
        model = RNN(p)
    elif p.model == 'lstm':
        model = LSTM(p)
    elif p.model == 'gru':
        model = GRU(p)
    elif p.model == 'bert':
        model = BERT(p)
    else:
        raise ValueError(f'Model {p.model} not recognized')

    print(f'{p.model} was created\n')
    
    return model


class LSTM(nn.Module):
  def __init__(self, p):
    
    super().__init__()
    
    self.bidirectional = p.bidirectional
    
    self.rnn = nn.LSTM(input_size=p.input_dim, 
                      hidden_size=p.hidden_dim, 
                      num_layers=p.n_layers, 
                      bidirectional=p.bidirectional, 
                      dropout=p.dropout_p)
    
    fc_input_dim = 2*p.hidden_dim if self.bidirectional else p.hidden_dim
    self.fc = nn.Linear(fc_input_dim, p.output_dim)
    self.dropout = nn.Dropout(p.dropout_p)
    
  def forward(self, x):
    
    output, (hidden, cell) = self.rnn(x)

    if self.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    else:
      hidden = self.dropout(hidden[-1,:,:])
            
    # hidden = [batch size, hid dim * num directions]
        
    return torch.sigmoid(self.fc(hidden))

class Conv1D(nn.Module):

  def __init__(self, p):
    
    super().__init__()

    self.conv1 = nn.Conv1d(1, 16, 3)
    self.conv2 = nn.Conv1d(16, 32, 3)
    self.conv3 = nn.Conv1d(32, 64, 3)
    self.pool = nn.MaxPool1d(3, stride=2)

    self.fc = nn.Linear(18432,10)
    self.relu = nn.ReLU()
    
  def forward(self, x):
    
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)
    
    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)
    
    x = self.conv3(x)
    x = self.relu(x)

    x = x.view(-1)
    x = self.fc(x)
    
    return x
