import json
import torch
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
from metric_dreem import dreem_sleep_apnea_custom_metric


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def open_config_file(filepath):
    with open(filepath) as jsonfile:
        pdict = json.load(jsonfile)
        params = AttrDict(pdict)
    return params


def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs


def train_model(epoch, model, train_loader, optimizer, criterion, params):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    with tqdm(train_loader, 
              desc=(f'Train - Epoch: {epoch}'), 
              unit=' patient', 
              ncols=80, 
              unit_scale=params.batch_size) as t:

        for i, (signal, target) in enumerate(t):

            optimizer.zero_grad()
            signal = signal.type(torch.FloatTensor)
            signal, target = signal.cuda(), target.cuda()
            signal = signal.permute(1,0,2)

            preds = model(signal).squeeze(1)
            preds = preds.type(torch.FloatTensor).cpu()
            target = target.type(torch.FloatTensor).cpu()
            loss = criterion(preds, target)
            acc = dreem_sleep_apnea_custom_metric((preds.detach()>0.5).float(), target.detach())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)
    # return epoch_loss / len(train_loader)


def evaluate_model(epoch, model, val_loader, criterion, params):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with tqdm(val_loader, 
             desc=(f'Validation - Epoch: {epoch}'), 
             unit=' patient', 
             ncols=80, 
             unit_scale=params.batch_size) as t:

        with torch.no_grad():

            for i, (signal, target) in enumerate(t):

                signal = signal.type(torch.FloatTensor)
                signal, target = signal.cuda(), target.cuda()
                signal = signal.permute(1,0,2)

                preds = model(signal).squeeze(1)
                preds = preds.type(torch.FloatTensor).cpu()
                target = target.type(torch.FloatTensor).cpu()
                loss = criterion(preds, target)
                acc = dreem_sleep_apnea_custom_metric((preds.detach()>0.5).float(), target.detach())

                epoch_loss += loss.item()
                epoch_acc += acc

    return epoch_loss / len(val_loader), epoch_acc / len(val_loader)
    # return epoch_loss / len(val_loader)

def plot_curves(train_losses, train_accuracies, validation_losses, validation_accuracies, params):

    x = range(params.nepochs)
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(x, train_losses)
    # axs[0,0].set_title('Axis [0,0]')
    axs[0,0].set_xlabel('epochs')
    axs[0,0].set_ylabel('loss')
    axs[0,1].plot(x, validation_losses, 'tab:orange')
    # axs[0,1].set_title('Axis [0,1]')
    axs[0,1].set_xlabel('epochs')
    axs[0,1].set_ylabel('loss')
    axs[1,0].plot(x, train_accuracies, 'tab:green')
    # axs[1,0].set_title('Axis [1,0]')
    axs[1,0].set_xlabel('epochs')
    axs[1,0].set_ylabel('acc')
    axs[1,1].plot(x, validation_accuracies, 'tab:red')
    # axs[1,1].set_title('Axis [1,1]')
    axs[1,1].set_xlabel('epochs')
    axs[1,1].set_ylabel('acc')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('curves.pdf')