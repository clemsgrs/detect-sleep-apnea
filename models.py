import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel

allowed_model_names = ['conv', 'rnn', 'lstm', 'gru', 'transformer', 'encoder_decoder']

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
    x = x.squeeze(1)

    return x


class LSTM(nn.Module):
  def __init__(self, p):

    super().__init__()

    self.bidirectional = p.bidirectional

    self.rnn = nn.LSTM(input_size=p.input_dim,
                      hidden_size=p.hidden_dim,
                      num_layers=p.n_layers,
                      bidirectional=p.bidirectional,
                      dropout=p.dropout_p,
                      batch_first=True)

    conv_input_dim = 2*p.hidden_dim if self.bidirectional else p.hidden_dim
    self.conv = nn.Conv2d(1, 1, kernel_size=(1,conv_input_dim))
    self.fc = nn.Linear(in_features=conv_input_dim, out_features=1)
    self.dropout = nn.Dropout(p.dropout_p)

  def forward(self, x):

    # as batch_first = True, x is expected to be (batch, seq_len, input_size)
    # x = x.permute(1,0,2)
    x, (hidden, cell) = self.rnn(x)
    
    # x is (batch, seq_length, conv_input_dim)
    # if self.bidirectional:
    #   hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    # else:
    #   hidden = self.dropout(hidden[-1,:,:])

    # hidden = [batch size, hid dim * num directions]
    # x = x.permute(1,0,2)
    x = x.unsqueeze(1)
    x = self.conv(x)
    x = x.squeeze()

    return torch.sigmoid(x)


class BERT(nn.Module):

  def __init__(self, p):

    super().__init__()
    
    hidden_size = p.input_dim * len(p.signal_ids)
    self.bert_config = transformers.BertConfig(
      vocab_size=1, 
      hidden_size=hidden_size,
      num_hidden_layers=p.n_layers,
      num_attention_heads=p.n_heads,
      intermediate_size=p.ffn_dim,
      hidden_dropout_prob=p.dropout_p,
      attention_probs_dropout_prob=p.dropout_p,
      max_position_embeddings=p.seq_length,
      position_embedding_type='absolute'
    )

    self.bert = BertModel(self.bert_config)
    self.conv = nn.Conv2d(1, 1, kernel_size=(1,hidden_size))
    self.fc = nn.Linear(in_features=p.input_dim, out_features=1)
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.bert(inputs_embeds=x)
    x = x['last_hidden_state']
    x = x.unsqueeze(1)
    x = self.conv(x)
    x = x.squeeze()

    return torch.sigmoid(x)


class EncoderDecoder(nn.Module):

  def __init__(self, p):

    super().__init__()

    self.model = p.model
    self.seq_length = p.seq_length
    self.conv_output_dim = p.conv_output_dim
    self.relu = nn.ReLU()
    
    if p.encoder == 'conv2d':
      self.encoder = Conv2D(p)
    else:
      raise ValueError(f'{p.encoder} not supported yet')

    if p.decoder == 'lstm':
      p.input_dim = p.conv_output_dim
      self.decoder = LSTM(p)
    elif p.decoder == 'transformer':
      p.input_dim = p.conv_output_dim
      self.decoder = BERT(p)
    elif p.decoder != 'conv':
      raise ValueError(f'{p.model} not supported yet')

  def forward(self, x):

    x = self.encoder(x)
    x = self.relu(x)
    x = self.decoder(x)

    return x


def create_model(p):

    model = None
    if p.model not in allowed_model_names:
      raise ValueError(f'Model {p.model} not recognized')
    else:
      if p.model == 'encoder_decoder':
        model = EncoderDecoder(p)
        print(f'{p.model} was created: {p.encoder}+{p.decoder}\n')
      else:
        if p.model == 'lstm':
          model = LSTM(p)
        elif p.model == 'transformer':
          model = BERT(p)
        elif p.model == 'conv':
          model = Conv1D(p)
        print(f'{p.model} was created\n')
    
    return model