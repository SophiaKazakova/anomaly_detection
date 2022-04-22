# -*- coding: utf-8 -*-
"""DCASE_task2_AEOE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KE-lBAdr-3kkcrSZmjD0IcIXaKBc3U9c

##Baseline
"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import time

from sklearn.metrics import roc_auc_score
from sklearn import metrics

!wget 'https://zenodo.org/record/6355122/files/dev_fan.zip?download=1' -O fan.zip

!unzip fan.zip

!rm fan.zip

torch.manual_seed(24)
np.random.seed(24)

class FanDataset(Dataset):
  def __init__(self, data_path, use_augmentation = False, mode = 'train'):

    self.data_path = data_path
    self.use_augmentation = use_augmentation
    self.mode = mode

    print("Loading dataset index, mode = ", self.mode)
    self.files = []
    for filename in tqdm(os.listdir(self.data_path)):
      self.files.append(os.path.join(self.data_path, filename))
    
  def __len__(self):
    return len(self.files)

  def _preprocess(self, waveform):
    return waveform
  
  def _extract_embeddings(self, waveform, sr = 16000):
    transform = T.MelSpectrogram(sample_rate=sr,
                                 n_fft=1024,  # Num of points used to find FFT
                                 win_length=400,  # 25ms when sr 16kHz
                                 hop_length=160,  # 10ms when sr 16kHz
                                 n_mels=128,
                                 normalized=False,
                                 norm = None)
    mel_specgram = transform(waveform)
    return mel_specgram #Mel frequency spectrogram of size (…, n_mels, time). Remove axes of length one from mel_specgram

  def _post_processing(self, embedding):
    if self.use_augmentation == True:
      pass # we don't have any augmentation here

    return embedding

  def _extract_features(self, waveform, sr):
    waveform = self._preprocess(waveform)
    emb = self._extract_embeddings(waveform, sr)
    features = self._post_processing(emb)
    return features

  def _get_label(self, index):
    if self.mode == 'train':
      return 0
    else:
      if self.files[index].find('anomaly') == -1:
        return 0
      else:
        return 1

  def __getitem__(self, index):
    """
    This function load one file, and extract its features we are interested in.
    """
    waveform, sr = torchaudio.load(self.files[index], normalize=True)
    features = self._extract_features(waveform, sr)
    label = self._get_label(index)
    return features, label

train_data = FanDataset(data_path='/content/fan/train', use_augmentation=True, 
                       mode='train')
test_data = FanDataset(data_path='/content/fan/test', 
                      use_augmentation=False, mode='test')
# After we define the dataset, we can extract elements one by one without the 
# need to load all of them to the RAM at once!
# As an example I will get the first item in the dataset, and print its shape
features, label = train_data.__getitem__(0)
print("Features shape: ", features.shape)
print("Label: ", label)

train_loader = DataLoader(train_data, 
                          batch_size=32, 
                          shuffle=True, 
                          num_workers=2)
test_loader = DataLoader(test_data, 
                         #batch_size=None, 
                         shuffle=False, 
                         num_workers=2)

for features, labels in train_loader:
  print("Shape of batch of features:        ", features.shape)
  print("Shape of the corresponding labels: ", labels.shape)
  break

args = {
        'epochs': 30,
        'log_dir': os.path.join(os.getcwd(), 'logs'),
        'feature_extractor': 'mel_spectrogram',
        'data_root': os.path.join(os.getcwd(), 'dev_data'),
        'learning_rate': 0.001,
        'save': os.path.join(os.getcwd(), 'models')
    }

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from sklearn.metrics import roc_auc_score
from sklearn import metrics

class AE(torch.nn.Module):
  def __init__(self):
      super().__init__()
      
      self.l1 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l2 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l3 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l4 = nn.Sequential(
          nn.Linear(128,8),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l5 = nn.Sequential(
          nn.Linear(8,128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l6 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l7 = nn.Sequential(
          nn.Linear(128,128),
          nn.BatchNorm2d(1),
          nn.ReLU6(inplace=True))
      self.l8 = nn.Sequential(
          nn.Linear(128,128))

  def forward(self,x):
      x1 = self.l1(x)
      x2 = self.l2(x1)
      x3 = self.l3(x2)
      x4 = self.l4(x3)
      x5 = self.l5(x4)
      x6 = self.l6(x5)
      x7 = self.l7(x6)
      x8 = self.l8(x7)
      return x8

model = AE()

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = args['learning_rate'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_hist = []
for i in range(args['epochs']):
    begin_epoch = time.time()
    model.train()  # enter train mode
    loss_avg = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        data = torch.swapaxes(data,2,3)

        optimizer.zero_grad()

        # forward
        x = model(data)
        
        # backward
        loss = loss_function(x, data)
        loss.backward()
        optimizer.step()

        loss += loss.item()
    
    # compute the epoch training loss
    loss_avg = loss / len(train_loader)

    loss_hist.append(loss_avg)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, time {:.6f}".format(i + 1, args['epochs'], loss_avg, time.time() - begin_epoch))

torch.save(model.state_dict(), args['save'])

torch.save({
            'epoch': args['epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 'state_dict_model.pt')

y_pred = []
y_true = []

for data, target in test_loader:

    data = torch.swapaxes(data,2,3)

    model.eval()
    with torch.no_grad():
        input = data.float().to(device)
        output = model(input)

        loss = loss_function(output, input)

        y_pred.append(np.mean(loss.cpu().numpy()))
        y_true.append(target.item())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

auc = metrics.roc_auc_score(y_true, y_pred)
pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr = 0.1)
fpr, tpr, thresholds = metrics.roc_curve(1 + y_true, y_pred, pos_label=2)
#auc2 = metrics.auc(fpr, tpr)

print("class: fan")
print("AUC: ", auc)
#print("Anomaly acc 2: ", auc2)
print("pAUC: ", pauc)

"""## +OE

"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import time

from sklearn.metrics import roc_auc_score
from sklearn import metrics

import numpy as np

print (torch.__version__)

!wget 'https://zenodo.org/record/6355122/files/dev_fan.zip?download=1' -O fan.zip

!unzip fan.zip

!wget 'https://zenodo.org/record/6355122/files/dev_ToyCar.zip?download=1' -O ToyCar.zip

!wget 'https://zenodo.org/record/6355122/files/dev_bearing.zip?download=1' -O bearing.zip

!wget 'https://zenodo.org/record/6355122/files/dev_gearbox.zip?download=1' -O gearbox.zip

!wget 'https://zenodo.org/record/6355122/files/dev_slider.zip?download=1' -O slider.zip

!wget 'https://zenodo.org/record/6355122/files/dev_valve.zip?download=1' -O valve.zip

!wget 'https://zenodo.org/record/6355122/files/dev_ToyTrain.zip?download=1' -O ToyTrain.zip

#!unzip ToyCar.zip
#!unzip gearbox.zip
#!unzip bearing.zip
!unzip slider.zip
#!unzip ToyTrain.zip
#!unzip valve.nzip



!rm fan.zip toycar.zip bearing.zip slider.zip

torch.manual_seed(24)
np.random.seed(24)

class FanDataset(Dataset):
  def __init__(self, data_path, use_augmentation = False, mode = 'train'):

    self.data_path = data_path
    self.use_augmentation = use_augmentation
    self.mode = mode

    print("Loading dataset index, mode = ", self.mode)
    self.files = []
    for filename in tqdm(os.listdir(self.data_path)):
      self.files.append(os.path.join(self.data_path, filename))
    
  def __len__(self):
    return len(self.files)

  def _preprocess(self, waveform):
    return waveform
  
  def _extract_embeddings(self, waveform, sr = 16000):
    transform = T.MelSpectrogram(sample_rate=sr,
                                 n_fft=1024,  # Num of points used to find FFT
                                 win_length=400,  # 25ms when sr 16kHz
                                 hop_length=160,  # 10ms when sr 16kHz
                                 n_mels=128,
                                 normalized=False,
                                 norm = None)
    mel_specgram = transform(waveform)
    return mel_specgram #Mel frequency spectrogram of size (…, n_mels, time). Remove axes of length one from mel_specgram

  def _post_processing(self, embedding):
    if self.use_augmentation == True:
      pass # we don't have any augmentation here

    return embedding

  def _extract_features(self, waveform, sr):
    waveform = self._preprocess(waveform)
    emb = self._extract_embeddings(waveform, sr)
    features = self._post_processing(emb)
    return features

  def _get_label(self, index):
    if self.mode == 'train':
      return 0
    else:
      if self.files[index].find('anomaly') == -1:
        return 0
      else:
        return 1

  def __getitem__(self, index):
    """
    This function load one file, and extract its features we are interested in.
    """
    waveform, sr = torchaudio.load(self.files[index], normalize=True)
    features = self._extract_features(waveform, sr)
    label = self._get_label(index)
    return features, label

train_data_in = FanDataset(data_path='/content/fan/train', use_augmentation=False, 
                       mode='train')
train_data_out = FanDataset(data_path='/content/slider/train', use_augmentation=False, 
                       mode='train')
test_data = FanDataset(data_path='/content/fan/test', 
                      use_augmentation=False, mode='test')
# After we define the dataset, we can extract elements one by one without the 
# need to load all of them to the RAM at once!
# As an example I will get the first item in the dataset, and print its shape
features, label = train_data_out.__getitem__(0)
print("Features shape: ", features.shape)
print("Label: ", label)

train_loader_in = DataLoader(train_data_in, 
                          batch_size=32, 
                          shuffle=True, 
                          num_workers=2)
train_loader_out = DataLoader(train_data_out, 
                          batch_size=32, 
                          shuffle=True, 
                          num_workers=2)
test_loader = DataLoader(test_data, 
                         #batch_size=None, 
                         shuffle=False, 
                         num_workers=2)

for features, labels in train_loader_in:
  print("Shape of batch of features:        ", features.shape)
  print("Shape of the corresponding labels: ", labels.shape)
  break

args = {
        'epochs': 30,
        'log_dir': os.path.join(os.getcwd(), 'logs'),
        'feature_extractor': 'mel_spectrogram',
        'data_root': os.path.join(os.getcwd(), 'dev_data'),
        'learning_rate': 0.001,
        'save': os.path.join(os.getcwd(), 'models')
    }

class AE(torch.nn.Module):
  def __init__(self):
      super().__init__()
      
      self.l1 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l2 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l3 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l4 = nn.Sequential(
          nn.Linear(128,8),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l5 = nn.Sequential(
          nn.Linear(8,128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l6 = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU6(inplace=True),
          nn.BatchNorm2d(1))
      self.l7 = nn.Sequential(
          nn.Linear(128,128),
          nn.BatchNorm2d(1),
          nn.ReLU6(inplace=True))
      self.l8 = nn.Sequential(
          nn.Linear(128,128))

  def forward(self,x):
      x1 = self.l1(x)
      x2 = self.l2(x1)
      x3 = self.l3(x2)
      x4 = self.l4(x3)
      x5 = self.l5(x4)
      x6 = self.l6(x5)
      x7 = self.l7(x6)
      x8 = self.l8(x7)
      return x8

model = AE()

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = args['learning_rate'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_hist = []
for i in range(args['epochs']):
    begin_epoch = time.time()
    model.train()  # enter train mode
    loss_avg = 0.0

    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.to(device), target.to(device)

        data = torch.swapaxes(data,2,3)

        optimizer.zero_grad()

        # forward
        x = model(data)
        
        # backward
        loss = loss_function(x, data)
        loss.backward()
        optimizer.step()

        loss += loss.item()
    
    # compute the epoch training loss
    loss_avg = loss / (len(train_loader_in)+len(train_loader_out))

    loss_hist.append(loss_avg)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, time {:.6f}".format(i + 1, args['epochs'], loss_avg, time.time() - begin_epoch))

torch.save(model.state_dict(), args['save'])

torch.save({
            'epoch': args['epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 'state_dict_model_OE_FanGearbox.pt')

model.eval()

y_pred = []
y_true = []

for data, target in test_loader:

    data = torch.swapaxes(data,2,3)

    model.eval()
    with torch.no_grad():
        input = data.float().to(device)
        output = model(input)

        loss = loss_function(output, input)

        y_pred.append(np.mean(loss.cpu().numpy()))
        y_true.append(target.item())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

auc = metrics.roc_auc_score(y_true, y_pred)
pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr = 0.1)
fpr, tpr, thresholds = metrics.roc_curve(1 + y_true, y_pred, pos_label=2)
#auc2 = metrics.auc(fpr, tpr)

print("class: fan + slider")
print("AUC: ", auc)
#print("Anomaly acc 2: ", auc2)
print("pAUC: ", pauc)

"""## Результаты

class: fan

AUC:  0.5424444444444444

pAUC:  0.5029239766081871

class: fan + ToyCar

AUC:  0.5427000000000001

pAUC:  0.5026900584795322

class: fan + bearing

AUC:  0.5424

pAUC:  0.5025730994152047

class: fan + ToyTrain

AUC:  0.5461

pAUC:  0.5025146198830409

class: fan + gearbox

AUC:  0.542788888888889

pAUC:  0.501812865497076

class: fan + slider

AUC:  0.5512666666666667

pAUC:  0.5023976608187134
"""