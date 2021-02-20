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

    print(f'{p.model} was created')
    
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