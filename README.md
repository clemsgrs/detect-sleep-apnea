# Detect Sleep Apnea
ENS challenge data

# How to run code?
To train a model, just download the `dreem_lstm.ipynb` notebook and open it with Google Colab. In the notebook, we first download the data, then we run training.

# Roadmap:
- [ ] normalize raw inputs (substract mean & divide by std)
- [x] support a single signal input (tune `signal_id` in config file to chose which signal to use)
- [ ] support multiple signal input (8 signals in total)
- [x] add convolutions to process raw input before feeding them to the RNN / transformer model
- [x] try convolutions alone
- [ ] work on the hidden state sequence instead of the last hidden state
- [ ] smooth labels to add temporal information
- [ ] try better loss / training strategy to account for temporal relations in target prediction
- [ ] swap rnn for a self-attention encoder
- [ ] try training a decoder as in a seq2seq model
