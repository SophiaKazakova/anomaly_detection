import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn import metrics

torch.manual_seed(24)
np.random.seed(24)

args = {
    'epochs': 5,
    'log_dir': os.path.join(os.getcwd(), 'logs'),
    'feature_extractor': 'mel_spectrogram',
    'data_root': os.path.join(os.getcwd(), 'dev_data'),
    'learning_rate': 0.001,
    'save': os.path.join(os.getcwd(), 'models')
}

class FanDataset(Dataset):
    def __init__(self, data_path, use_augmentation=False, mode='train'):

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

    def _extract_embeddings(self, waveform, sr=16000):
        transform = T.MelSpectrogram(sample_rate=sr,
                                     n_fft=1024,  # Num of points used to find FFT
                                     win_length=400,  # 25ms when sr 16kHz
                                     hop_length=160,  # 10ms when sr 16kHz
                                     n_mels=128,
                                     normalized=False,
                                     norm=None)
        mel_specgram = transform(waveform)
        return mel_specgram

    def _post_processing(self, embedding):
        if self.use_augmentation == True:
            pass  # we don't have any augmentation here

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


train_data_in = FanDataset(data_path='./dev_data/fan/train',
                           use_augmentation=True, mode='train')
train_data_out = FanDataset(data_path='./dev_data/slider/train',
                            use_augmentation=False, mode='train')
test_data_00 = FanDataset(data_path='./dev_data/fan/test/sec00', use_augmentation=False, mode='test')
test_data_01 = FanDataset(data_path='./dev_data/fan/test/sec01', use_augmentation=False, mode='test')
test_data_02 = FanDataset(data_path='./dev_data/fan/test/sec02', use_augmentation=False, mode='test')

train_loader_in = DataLoader(train_data_in, batch_size=32, shuffle=True, num_workers=2)
train_loader_out = DataLoader(train_data_out, batch_size=32, shuffle=True, num_workers=2)
test_loader_00 = DataLoader(test_data_00, shuffle=False, num_workers=2)
test_loader_01 = DataLoader(test_data_01, shuffle=False, num_workers=2)
test_loader_02 = DataLoader(test_data_02, shuffle=False, num_workers=2)


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
            nn.Linear(128, 8),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(1))
        self.l5 = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(1))
        self.l6 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(1))
        self.l7 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm2d(1),
            nn.ReLU6(inplace=True))
        self.l8 = nn.Sequential(
            nn.Linear(128, 128))

    def forward(self, x):
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
optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_hist = []

# --------------------------------------- TRAINING -------------------------------------------
min_valid_loss = np.inf

for i in range(args['epochs']):
    train_loss = 0.0
    begin_epoch = time.time()
    model.train()  # enter train mode

    for data, target in train_loader_in:
        data, target = data.to(device), target.to(device)

        data = torch.swapaxes(data, 2, 3)

        optimizer.zero_grad()
        x = model(data)
        loss = loss_function(x, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for data, target in test_loader_00:
        data, target = data.to(device), target.to(device)

        data = torch.swapaxes(data, 2, 3)

        x = model(data)
        loss = loss_function(x, data)
        valid_loss = loss.item() * data.sise(0)

    print(
        f'Epoch {i + 1} \t\t Training Loss: {train_loss / len(train_loader_in)} \t\t Validation Loss: {valid_loss / len(test_loader_00)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

y_pred00 = []
y_true00 = []
AUCavg = 0
pAUCavg = 0

for data, target in test_loader_00:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input00 = data.float().to(device)
        output = model(input00)

        loss = loss_function(output, input00)

        y_pred00.append(np.mean(loss.cpu().numpy()))
        y_true00.append(target.item())

y_pred00 = np.array(y_pred00)
y_true00 = np.array(y_true00)

AUC = metrics.roc_auc_score(y_true00, y_pred00)
pAUC = metrics.roc_auc_score(y_true00, y_pred00, max_fpr=0.1)

AUCavg += AUC
pAUCavg += pAUC

print("class: fan on pure AE section 00")
print("AUC: ", AUC)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

y_pred01 = []
y_true01 = []

for data, target in test_loader_01:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input01 = data.float().to(device)
        output = model(input01)

        loss = loss_function(output, input01)

        y_pred01.append(np.mean(loss.cpu().numpy()))
        y_true01.append(target.item())

y_pred = np.array(y_pred01)
y_true = np.array(y_true01)

AUC = metrics.roc_auc_score(y_true01, y_pred01)
pAUC = metrics.roc_auc_score(y_true01, y_pred01, max_fpr=0.1)

AUCavg += AUC
pAUCavg += pAUC

print("class: fan on pure AE section 01")
print("AUC: ", AUC)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

y_pred02 = []
y_true02 = []

for data, target in test_loader_02:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input02 = data.float().to(device)
        output = model(input02)

        loss = loss_function(output, input02)

        y_pred02.append(np.mean(loss.cpu().numpy()))
        y_true02.append(target.item())

y_pred02 = np.array(y_pred)
y_true02 = np.array(y_true)

AUC = metrics.roc_auc_score(y_true02, y_pred02)
pAUC = metrics.roc_auc_score(y_true02, y_pred02, max_fpr=0.1)

AUCavg += AUC
pAUCavg += pAUC

print("class: fan on pure AE section 02")
print("AUC: ", AUC)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

print("AUC avg:", AUCavg/3)
print("pAUC avg:", pAUCavg/3)

# --------------------------------------------- OE -------------------------------------
for i in range(args['epochs']):
    begin_epoch = time.time()
    model.train()  # enter train mode
    loss_avg = 0.0

    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.to(device), target.to(device)

        data = torch.swapaxes(data, 2, 3)

        optimizer.zero_grad()

        # forward
        x = model(data)

        # backward
        loss = loss_function(x, data)
        loss.backward()
        optimizer.step()

        loss += loss.item()

    # compute the epoch training loss
    loss_avg = loss / (len(train_loader_in) + len(train_loader_out))

    loss_hist.append(loss_avg)

    # display the epoch training loss
    print(
        "epoch : {}/{}, loss = {:.6f}, time {:.6f}".format(i + 1, args['epochs'], loss_avg, time.time() - begin_epoch))


torch.save({
    'epoch': args['epochs'],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss}, 'state_dict_model_OE_FanSlider.pt')

model.eval()

y_pred00 = []
y_true00 = []
AUCavg = 0
pAUCavg = 0

for data, target in test_loader_00:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input00 = data.float().to(device)
        output = model(input00)

        loss = loss_function(output, input00)

        y_pred00.append(np.mean(loss.cpu().numpy()))
        y_true00.append(target.item())

y_pred00 = np.array(y_pred00)
y_true00 = np.array(y_true00)

AUC = metrics.roc_auc_score(y_true00, y_pred00)
pAUC = metrics.roc_auc_score(y_true00, y_pred00, max_fpr=0.1)

AUCavg += AUC
pAUCavg += pAUC

print("class: fan on OE + slider section 00")
print("AUC: ", AUC)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

y_pred01 = []
y_true01 = []

for data, target in test_loader_01:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input01 = data.float().to(device)
        output = model(input01)

        loss = loss_function(output, input01)

        y_pred01.append(np.mean(loss.cpu().numpy()))
        y_true01.append(target.item())

y_pred = np.array(y_pred01)
y_true = np.array(y_true01)

AUC = metrics.roc_auc_score(y_true01, y_pred01)
pAUC = metrics.roc_auc_score(y_true01, y_pred01, max_fpr=0.1)

AUCavg += AUC
pAUCavg += pAUC

print("class: fan on OE + slider section 01")
print("AUC: ", AUC)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

y_pred02 = []
y_true02 = []

for data, target in test_loader_02:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input02 = data.float().to(device)
        output = model(input02)

        loss = loss_function(output, input02)

        y_pred02.append(np.mean(loss.cpu().numpy()))
        y_true02.append(target.item())

y_pred02 = np.array(y_pred)
y_true02 = np.array(y_true)

AUC = metrics.roc_auc_score(y_true02, y_pred02)
pAUC = metrics.roc_auc_score(y_true02, y_pred02, max_fpr=0.1)

AUCavg += AUC
pAUCavg += pAUC

print("class: fan on OE + slider section 02")
print("AUC: ", AUC)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

print("AUC avg:", AUCavg/3)
print("pAUC avg:", pAUCavg/3)