import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import statistics

from tqdm import tqdm

from sklearn import metrics

torch.manual_seed(24)
np.random.seed(24)

args = {
    'epochs': 20,
    'log_dir': os.path.join(os.getcwd(), 'logs'),
    'feature_extractor': 'mel_spectrogram',
    'data_root': os.path.join(os.getcwd(), 'dev_data'),
    'learning_rate': 0.001,
    'save': os.path.join(os.getcwd(), 'models'),
    'target_class': 'valve',
    'OE': 'slider'  # fan/slider/ToyCar/ToyTrain/valve/gearbox/bearing
}

class MachineDataset(Dataset):
    def __init__(self, data_path, eval_path, use_augmentation=False, mode='train'):

        self.data_path = data_path
        self.eval_path = eval_path
        self.use_augmentation = use_augmentation
        self.mode = mode

        print("Loading dataset index, mode = ", self.mode)
        self.files = []
        for filename in tqdm(os.listdir(self.data_path)):
            self.files.append(os.path.join(self.data_path, filename))

        if data_path != eval_path:
            for filename in tqdm(os.listdir(self.eval_path)):
                self.files.append(os.path.join(self.eval_path, filename))

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


print('Loading data...')

target_path = os.path.join('./dev_data/', args['target_class'])
eval_path = os.path.join('./dev_data/', args['target_class'], 'eval')
OE_path = os.path.join('./dev_data/', args['OE'], 'train')

train_data_in = MachineDataset(data_path=os.path.join(target_path, 'train'), eval_path = eval_path,
                               use_augmentation=False, mode='train')
train_data_out = MachineDataset(data_path = OE_path, eval_path = OE_path,
                                use_augmentation=False, mode='train')

train_size = int(0.8 * len(train_data_in))
eval_size = len(train_data_in) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(train_data_in, [train_size, eval_size])

test_data_s00 = MachineDataset(data_path=os.path.join(target_path, 'test/sec00/source'), eval_path=os.path.join(target_path, 'test/sec00/source'), use_augmentation=False, mode='test')
test_data_s01 = MachineDataset(data_path=os.path.join(target_path, 'test/sec01/source'), eval_path=os.path.join(target_path, 'test/sec01/source'), use_augmentation=False, mode='test')
test_data_s02 = MachineDataset(data_path=os.path.join(target_path, 'test/sec02/source'), eval_path=os.path.join(target_path, 'test/sec02/source'), use_augmentation=False, mode='test')

test_data_t00 = MachineDataset(data_path=os.path.join(target_path, 'test/sec00/target'), eval_path=os.path.join(target_path, 'test/sec00/target'), use_augmentation=False, mode='test')
test_data_t01 = MachineDataset(data_path=os.path.join(target_path, 'test/sec01/target'), eval_path=os.path.join(target_path, 'test/sec01/target'), use_augmentation=False, mode='test')
test_data_t02 = MachineDataset(data_path=os.path.join(target_path, 'test/sec02/target'), eval_path=os.path.join(target_path, 'test/sec02/target'), use_augmentation=False, mode='test')

train_loader_in = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
train_loader_out = DataLoader(train_data_out, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True, num_workers=2)

test_loader_s00 = DataLoader(test_data_s00, shuffle=False, num_workers=2)
test_loader_s01 = DataLoader(test_data_s01, shuffle=False, num_workers=2)
test_loader_s02 = DataLoader(test_data_s02, shuffle=False, num_workers=2)

test_loader_t00 = DataLoader(test_data_t00, shuffle=False, num_workers=2)
test_loader_t01 = DataLoader(test_data_t01, shuffle=False, num_workers=2)
test_loader_t02 = DataLoader(test_data_t02, shuffle=False, num_workers=2)

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
'''
print('Starting training...')
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
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        data = torch.swapaxes(data, 2, 3)

        x = model(data)
        loss = loss_function(x, data)
        valid_loss = loss.item()

    print(
        f'Epoch {i + 1} \t\t Training Loss: {train_loss / len(train_loader_in)} \t\t Validation Loss: {valid_loss}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

y_pred00 = []
y_true00 = []
AUCavg = 0
pAUCavg = 0

model.load_state_dict(torch.load('saved_model.pth'))

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

y_pred01 = np.array(y_pred01)
y_true01 = np.array(y_true01)

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

y_pred02 = np.array(y_pred02)
y_true02 = np.array(y_true02)

AUC = metrics.roc_auc_score(y_true02, y_pred02)
pAUC = metrics.roc_auc_score(y_true02, y_pred02, max_fpr=0.1)

AUCavg += AUC
pAUCavg += pAUC

print("class: fan on pure AE section 02")
print("AUC: ", AUC)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

print("AUC avg:", AUCavg/3)
print("pAUC avg:", pAUCavg/3)'''
# --------------------------------------------- OE -------------------------------------
min_valid_loss = np.inf

print('Starting ', args['target_class'], ' OE training, OE class is', args['OE'])
for i in range(args['epochs']):
    train_loss = 0.0
    begin_epoch = time.time()
    model.train()

    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
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

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        data = torch.swapaxes(data, 2, 3)
        x = model(data)
        loss = loss_function(x, data)
        valid_loss = loss.item()

    model_name = 'saved_OE_model_' + args['target_class']

    print(
        f'Epoch {i + 1} \t\t Training Loss: {train_loss / len(train_loader_in)} \t\t Validation Loss: {valid_loss}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), model_name)

y_predoe00s = []
y_trueoe00s = []
AUCavg_s = 0
AUCavg_t = 0
pAUCavg = 0

AUC_list_s = []
AUC_list_t = []
pAUC_list = []

model.load_state_dict(torch.load(model_name))

for data, target in test_loader_s00:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input00 = data.float().to(device)
        output = model(input00)

        loss = loss_function(output, input00)

        y_predoe00s.append(np.square(loss.cpu().numpy()))
        y_trueoe00s.append(target.item())

y_predoe00s = np.array(y_predoe00s)
y_trueoe00s = np.array(y_trueoe00s)

AUC_s = metrics.roc_auc_score(y_trueoe00s, y_predoe00s)
pAUC = metrics.roc_auc_score(y_trueoe00s, y_predoe00s, max_fpr=0.1)

AUCavg_s += AUC_s
pAUCavg += pAUC

AUC_list_s.append(AUC_s)
pAUC_list.append(pAUC)

print("class: OE", args['target_class'], "section 00, outlier type: ", args['OE'], 'source')
print("AUC: ", AUC_s)
print("pAUC: ", pAUC)

y_predoe00t = []
y_trueoe00t = []

model.load_state_dict(torch.load('saved_OE_model.pth'))

for data, target in test_loader_t00:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input00 = data.float().to(device)
        output = model(input00)

        loss = loss_function(output, input00)

        y_predoe00t.append(np.mean(loss.cpu().numpy()))
        y_trueoe00t.append(target.item())

y_predoe00t = np.array(y_predoe00t)
y_trueoe00t = np.array(y_trueoe00t)

AUC_t = metrics.roc_auc_score(y_trueoe00t, y_predoe00t)
pAUC = metrics.roc_auc_score(y_trueoe00t, y_predoe00t, max_fpr=0.1)

AUCavg_t += AUC_t
pAUCavg += pAUC

AUC_list_t.append(AUC_t)
pAUC_list.append(pAUC)

print("class: OE", args['target_class'], "section 00, outlier type: ", args['OE'], 'target')
print("AUC: ", AUC_t)
print("pAUC: ", pAUC)

y_predoe01s = []
y_trueoe01s = []

for data, target in test_loader_s01:
    data = torch.swapaxes(data, 2, 3)


    model.eval()
    with torch.no_grad():
        input01 = data.float().to(device)
        output = model(input01)

        loss = loss_function(output, input01)

        y_predoe01s.append(np.mean(loss.cpu().numpy()))
        y_trueoe01s.append(target.item())

y_predoe01s = np.array(y_predoe01s)
y_trueoe01s = np.array(y_trueoe01s)

AUC_s = metrics.roc_auc_score(y_trueoe01s, y_predoe01s)
pAUC = metrics.roc_auc_score(y_trueoe01s, y_predoe01s, max_fpr=0.1)

AUCavg_s += AUC_s
pAUCavg += pAUC

AUC_list_s.append(AUC_s)
pAUC_list.append(pAUC)

print("class: OE", args['target_class'], "section 01, outlier type: ", args['OE'], 'source')
print("AUC: ", AUC_s)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)

y_predoe01t = []
y_trueoe01t = []

for data, target in test_loader_t01:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input01 = data.float().to(device)
        output = model(input01)

        loss = loss_function(output, input01)

        y_predoe01t.append(np.mean(loss.cpu().numpy()))
        y_trueoe01t.append(target.item())

y_predoe01t = np.array(y_predoe01t)
y_trueoe01t = np.array(y_trueoe01t)

AUC_t = metrics.roc_auc_score(y_trueoe01t, y_predoe01t)
pAUC = metrics.roc_auc_score(y_trueoe01t, y_predoe01t, max_fpr=0.1)

AUCavg_t += AUC_t
pAUCavg += pAUC

AUC_list_t.append(AUC_t)
pAUC_list.append(pAUC)

print("class: OE", args['target_class'], "section 01, outlier type: ", args['OE'], 'target')
print("AUC: ", AUC_t)
# print("Anomaly acc 2: ", auc2)
print("pAUC: ", pAUC)


y_predoe02s = []
y_trueoe02s = []

for data, target in test_loader_s02:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input02 = data.float().to(device)
        output = model(input02)

        loss = loss_function(output, input02)

        y_predoe02s.append(np.mean(loss.cpu().numpy()))
        y_trueoe02s.append(target.item())

y_predoe02s = np.array(y_predoe02s)
y_trueoe02s = np.array(y_trueoe02s)

AUC_s = metrics.roc_auc_score(y_trueoe02s, y_predoe02s)
pAUC = metrics.roc_auc_score(y_trueoe02s, y_predoe02s, max_fpr=0.1)

AUCavg_s += AUC_s
pAUCavg += pAUC

AUC_list_s.append(AUC_s)
pAUC_list.append(pAUC)

print("class: OE", args['target_class'], "section 02, outlier type: ", args['OE'], 'source')
print("AUC: ", AUC_s)
print("pAUC: ", pAUC)

y_predoe02t = []
y_trueoe02t = []

for data, target in test_loader_t02:
    data = torch.swapaxes(data, 2, 3)

    model.eval()
    with torch.no_grad():
        input02 = data.float().to(device)
        output = model(input02)

        loss = loss_function(output, input02)

        y_predoe02t.append(np.mean(loss.cpu().numpy()))
        y_trueoe02t.append(target.item())

y_predoe02t = np.array(y_predoe02t)
y_trueoe02t = np.array(y_trueoe02t)

AUC_t = metrics.roc_auc_score(y_trueoe02t, y_predoe02t)
pAUC = metrics.roc_auc_score(y_trueoe02t, y_predoe02t, max_fpr=0.1)

AUCavg_t += AUC_t
pAUCavg += pAUC

AUC_list_t.append(AUC_t)
pAUC_list.append(pAUC)

print("class: OE", args['target_class'], "section 02, outlier type: ", args['OE'], 'target')
print("AUC: ", AUC_t)
print("pAUC: ", pAUC)


print("AUC_s avg:", AUCavg_s/3)
print("AUC_t avg:", AUCavg_t/3)
print("pAUC avg:", pAUCavg/6)

AUC_harmonic_s = statistics.harmonic_mean(AUC_list_s)
AUC_harmonic_t = statistics.harmonic_mean(AUC_list_t)
pAUC_harmonic = statistics.harmonic_mean(pAUC_list)


print("AUC harmonic source:", AUC_harmonic_s)
print("AUC harmonic target:", AUC_harmonic_t)

print("pAUC harmonic:", pAUC_harmonic)
