import torch
import torch.nn as nn
import torch.nn.functional as F

allowed_model_names = ['conv', 'rnn', 'lstm', 'gru', 'transformer']

def create_model(p):

    model = None
    if p.model not in allowed_model_names:
      raise ValueError(f'Model {p.model} not recognized')
    else:
      if p.use_conv:
        model = CustomModel(p)
      else:
        if p.model == 'lstm':
          model = LSTM(p)
        elif p.model == 'conv':
          model = Conv1D(p)

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

    x = x.permute(1,0,2)
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
    self.maxpool = nn.MaxPool1d(10)
    self.avgpool = nn.AdaptiveAvgPool1d(100)

    self.fc = nn.Linear(100*64, 90*p.conv_output_dim)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()

  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.avgpool(x)

    x = self.flatten(x)
    x = self.fc(x)
    x = x.reshape(x.shape[0], 90, 10)

    return x

class CustomModel(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.conv = Conv1D(p)
    if p.model == 'lstm':
      p.input_dim = p.conv_output_dim
      self.rnn = LSTM(p)
    else:
      raise ValueError(f'{p.model} not supported yet')
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.conv(x)
    x = self.relu(x)
    x = self.rnn(x)

    return x
