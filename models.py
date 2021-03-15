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

    # x should be (seq_len, batch, input_size)
    x = x.permute(1,0,2)
    output, (hidden, cell) = self.rnn(x)

    if self.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    else:
      hidden = self.dropout(hidden[-1,:,:])

    # hidden = [batch size, hid dim * num directions]

    return self.fc(hidden)

class Conv1D(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.seq_length = p.seq_length
    self.conv_output_dim = p.conv_output_dim
    self.in_channels = len(p.signal_ids)
    self.ks = p.kernel_sizes
    self.s = p.strides
    self.use_maxpool = p.use_maxpool
    self.use_avgpool = p.use_avgpool
    assert len(self.ks) == 3

    self.conv1 = nn.Conv1d(self.in_channels, 16, self.ks[0], self.s[0])
    self.conv2 = nn.Conv1d(16, 32, self.ks[1], self.s[1])
    self.conv3 = nn.Conv1d(32, 64, self.ks[2], self.s[2])
    
    self.maxpool = nn.MaxPool1d(10)
    self.avgpool = nn.AdaptiveAvgPool1d(100)
    self.dropout = nn.Dropout(p.dropout_p)

    self.fc = nn.Linear(100*64, self.seq_length*self.conv_output_dim)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()

  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.dropout(x)
    if self.use_maxpool:
      x = self.maxpool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.dropout(x)
    if self.use_maxpool:
      x = self.maxpool(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.dropout(x)
    if self.use_avgpool:
      x = self.avgpool(x)

    x = self.flatten(x)
    x = self.fc(x)

    return x


class Conv2D(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.seq_length = p.seq_length
    self.conv_output_dim = p.conv_output_dim
    self.in_channels = len(p.signal_ids)
    self.ks = p.kernel_sizes
    self.s = p.strides
    self.use_maxpool = p.use_maxpool
    self.use_avgpool = p.use_avgpool
    assert len(self.ks) == 3

    self.conv1 = nn.Conv2d(self.in_channels, 4, kernel_size=(1,50))
    self.conv2 = nn.Conv2d(4, 8, kernel_size=(1,25))
    self.conv3 = nn.Conv2d(8, 16, kernel_size=(1,13))
    self.conv4 = nn.Conv2d(16, 1, kernel_size=(1,1))
    
    self.maxpool = nn.MaxPool2d(kernel_size=(1,2))
    self.avgpool = nn.AdaptiveAvgPool2d(100)
    self.dropout = nn.Dropout(p.dropout_p)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.conv4(x)
    x = self.relu(x)

    # for good RNN integration, x should be (batch, seq_len, conv_output_dim)
    x = x.squeeze()

    return x

class CustomModel(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.model = p.model
    self.seq_length = p.seq_length
    self.conv_output_dim = p.conv_output_dim
    self.conv = Conv2D(p)
    self.relu = nn.ReLU()

    if p.model == 'lstm':
      p.input_dim = p.conv_output_dim
      self.rnn = LSTM(p)
    elif p.model != 'conv':
      raise ValueError(f'{p.model} not supported yet')

  def forward(self, x):

    x = self.conv(x)
    
    if self.model == 'lstm':
      # x = x.reshape(x.shape[0], self.seq_length, self.conv_output_dim)
      x = self.relu(x)
      x = self.rnn(x)

    return torch.sigmoid(x)
