# Detect Sleep Apnea
ENS data challenge submitted by [Dreem](https://dreem.com) where the goal was to detect sleep apnea events from polysomnography (PSG) signals.<br>
This challenge was proposed as part of the "Sparse Representation" course dispensed by Pr. S. Mallat at MVA.<br>
We ended with a **0.6503** score in the private academic leader-board in the limited time of that course (rank 1).

Our work is described in [this]() scientific report.

# How to run code?
To train a model, just download the `dreem_lstm.ipynb` notebook and open it with Google Colab. In the notebook, we first download the data, then we run training.

# Roadmap:
- [x] normalize raw inputs (substract mean & divide by std) (done sample-wise, may be better to do it signal-wise)
- [x] support a single signal input (tune `signal_id` in config file to chose which signal to use)
- [x] support multiple signal input (8 signals in total)
- [x] add convolutions to process raw input before feeding them to the RNN / transformer model
- [x] try convolutions alone
- [x] work on the hidden state sequence instead of the last hidden state
- [ ] smooth labels to add temporal information
- [ ] try better loss / training strategy to account for temporal relations in target prediction
- [x] try self-attention encoder instead of rnn (couldn't manage to train the self-att encoder properly...)
- [ ] try training a decoder as in a seq2seq model
- [ ] try different classification thresholds
- [ ] try connexity in forward
- [ ] try connexity as post-processing (if statement everywhere)
- [ ] try smoothing with bigger sigma (sigma = 2 good start)
