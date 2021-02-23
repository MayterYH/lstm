import torch
import torch.nn as nn
from torch.optim import *

import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import random


device = torch.device('cuda:0')

df_data = pd.read_csv(r'./data/ssn_mon.csv', header=0, index_col=0)

all_data = df_data.iloc[:, 1].values

info = pd.read_csv(r'./picture/log.txt',header=None,index_col=None,sep=':')
info = list(info.iloc[-9:,1])
m_time=str(info[1][1:])
m_lr=float(info[2])
m_layer=int(info[3])
m_cell=int(info[4])
m_test=int(info[5])
m_window=int(info[6])
m_epoch=int(info[7])

# batch = 12
lr=m_lr
num_layer=m_layer
hidden_layer_set=m_cell
test_data_size=m_test
train_window=m_window
epochs=m_epoch

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)




def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(train_data_normalized, train_window)



class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=hidden_layer_set, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layer,dropout=0.5)

        # self.linear_h = nn.Linear(hidden_layer_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(num_layer, 1, self.hidden_layer_size).to(device),
                            torch.zeros(num_layer, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # predictions = self.linear(predictions_h)
        return predictions[-1]


model = torch.load('./model/%s.pt' % m_time)

# time_now=datetime.datetime.now().isoformat().replace(':', '-')[:16]
time_now=m_time
fut_pred = test_data_size

test_inputs = train_data_normalized[-train_window:].tolist()

model.eval().to(device)

for i in tqdm(range(fut_pred)):
    seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(num_layer, 1, model.hidden_layer_size).to(device),
                        torch.zeros(num_layer, 1, model.hidden_layer_size).to(device))
        test_inputs.append(model(seq).item())

true_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1)).T[0]

"""
检验预测效果
"""
future_set=120
test_data_normalized = scaler.fit_transform(test_data.reshape(-1, 1))
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)

future_input=test_data_normalized[-train_window-future_set:].tolist()[:-future_set]
for j in tqdm(range(future_set)):
    seq = torch.FloatTensor(future_input[-train_window:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(num_layer, 1, model.hidden_layer_size).to(device),
                        torch.zeros(num_layer, 1, model.hidden_layer_size).to(device))
        future_input.append(model(seq).item())
future_predictions = scaler.inverse_transform(np.array(future_input).reshape(-1, 1)).T[0]


"""
预测未来
"""
future_set_t=120
test_data_normalized = scaler.fit_transform(test_data.reshape(-1, 1))
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)

future_input_t=test_data_normalized[-train_window:].tolist()
for j in tqdm(range(future_set_t)):
    seq = torch.FloatTensor(future_input_t[-train_window:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(num_layer, 1, model.hidden_layer_size).to(device),
                        torch.zeros(num_layer, 1, model.hidden_layer_size).to(device))
        future_input_t.append(model(seq).item())
future_predictions_t = scaler.inverse_transform(np.array(future_input_t).reshape(-1, 1)).T[0]

pd.DataFrame(future_predictions_t)


"""
测试集全检验
"""
# print(train_data[-train_window:])
# print(test_data)

test_data_plus=np.append(train_data[-train_window:],test_data)
# test_data_plus.extend(test_data)
test_data_normalized_p = scaler.fit_transform(test_data_plus.reshape(-1, 1))
test_inout_seq = create_inout_sequences(test_data_normalized_p, train_window)
# print(len(test_inout_seq))
test_sig_pre=[]
for seq,_ in tqdm(test_inout_seq):
    # print(seq)
    seq=torch.FloatTensor(seq).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(num_layer, 1, model.hidden_layer_size).to(device),
                        torch.zeros(num_layer, 1, model.hidden_layer_size).to(device))
        test_sig_pre.append(model(seq).item())
future_predictions_all = scaler.inverse_transform(np.array(test_sig_pre).reshape(-1, 1)).T[0]



# print(true_predictions[0])
with open("./forecast/log.txt", "a") as f:
    print('parameters:\ntime: %s\nlr: %.4f\nnum_layer: %d\nhidden_layer_set: %d'
          '\ntest_size: %d\ntrain_windous: %d\nepochs: %d' %
          (time_now, lr, num_layer,
           hidden_layer_set, test_data_size, train_window, epochs), file=f)
with open("./forecast/log.txt", "a") as f:
    print('====================================', file=f)
print('Over!')
# pd.DataFrame(loss_info).to_csv(r'loss.csv')
#%%
fig,ax=plt.subplots(figsize=(10, 4), dpi=250)
plt.style.use('fast')
x_time = [i for i in range(len(all_data))][2000:]
x_ticks = [list(df_data.index)[n][:-6] for n in range(len(all_data))][2000:]

x_time_t=[x_time[-1]+1+i for i in range(future_set_t)]

ax.plot(x_time[:-test_data_size], all_data[2000:-test_data_size], lw=1.5, c='dodgerblue', label='Train')

ax.plot(x_time[-test_data_size:], all_data[-test_data_size:], lw=1.5, c='r', label='Test')

ax.plot(x_time[-test_data_size:], future_predictions_all,ls='--', lw=1, c='k', label='Pred Short',zorder=10)


ax.plot(x_time[-test_data_size:], true_predictions[-test_data_size:], ls='--', lw=1.5, c='m', label='Pred Long')
ax.plot(x_time[-future_set:], future_predictions[-future_set:], ls='--', lw=1.5, c='b', label='Test Future')
ax.plot(x_time_t, future_predictions_t[-future_set_t:], ls='--', lw=1.5, c='c', label='Future')


ax1 = ax.inset_axes([0.82, 0.63, 0.15, 0.3])
ax1.plot(x_time[-test_data_size:], all_data[-test_data_size:], lw=1.5, c='r', label='Test')
ax1.plot(x_time[-test_data_size:], future_predictions_all,ls='--', lw=1, c='k', label='Test Short',zorder=10)
ax1.plot(x_time[-future_set:], future_predictions[-future_set:], ls='--', lw=1.5, c='b', label='Future')
ax1.plot(x_time_t, future_predictions_t[-future_set_t:], ls='--', lw=1.5, c='c', label='Future')
ax1.set_xlim(x_time[-80],x_time[-1]+30+future_set_t)
# ax1.set_xtickslabels(label=['2007','2017'])
ax1.set_ylim(0,80)
ax1.set_xticks([1300,])
ax1.set_xticklabels(['2007'])
ax1.tick_params(axis='both', direction='in', which='both')

ax.set_title('LSTM-Parameters: lr:%.4f num_layer:%d hidden_layer_set:%d test_size:%d train_windous:%d epochs:%d' % (
    lr, num_layer, hidden_layer_set, test_data_size, train_window, epochs), fontsize=6, weight='bold', style='italic')

ax.set_xlabel('Time', fontsize=10, weight='bold', style='italic')
ax.set_ylabel('SSN', fontsize=10, weight='bold', style='italic')
ax.axvline(x=x_time[-test_data_size], lw=1, ls='-.', c='k', alpha=0.4)
ax.axvline(x=x_time[-1], lw=1, ls='-.', c='k', alpha=0.4)
plt.xticks([x_time[m * 120 + 10] for m in range(11)],
           labels=[x_ticks[m * 120 + 10] for m in range(11)], size=8, weight='bold')
plt.yticks([0, 50, 100, 150, 200, 250], size=8, weight='bold')
plt.tick_params(axis='both', direction='in', which='both')
ax.legend(loc=2,fontsize=6)
ax.indicate_inset_zoom(ax1)
plt.savefig(r"./forecast/%s.png" % time_now)
# plt.show()
# plt.figure(figsize=(8, 4), dpi=250)
# plt.style.use('fast')
# plt.title('LSTM-Parameters: lr:%.4f num_layer:%d hidden_layer_set:%d test_size:%d train_windous:%d epochs:%d' % (
#     lr, num_layer, hidden_layer_set, test_data_size, train_window, epochs), fontsize=6, weight='bold', style='italic')
# plt.ylabel('SSN')
# plt.grid(True)
# plt.autoscale(axis='x', tight=True)
# plt.plot(range(len(all_data)), all_data,c='dodgerblue')
# plt.plot([i for i in range(len(all_data))][-len(actual_predictions):], actual_predictions,c='m')
# plt.tick_params(axis='both', direction='in', which='both')
# plt.savefig(r'./pic/Stock/Stock-%s.png' % time_now)
# plt.show()
# plt.figure(figsize=(8,5),dpi=200)
# plt.style.use('fivethirtyeight')
# # plt.plot(range(len(loss_info)),loss_info,lw=2, c='grey', label='LOSS')
# plt.tick_params(axis='both', direction='in', which='both')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# ax.legend(loc="upper left")
# plt.savefig(r"./picture/Loss-%s.png" % time_now)

plt.figure(figsize=(8,5),dpi=200)
plt.style.use('fivethirtyeight')
err = abs(future_predictions[-future_set:]-test_data[-future_set:])/test_data[-future_set:]
plt.plot(range(len(err)),err,lw=2, c='grey', label='RE')
plt.xlabel('Months')
plt.ylabel('Erroor')
ax.legend(loc="upper left")
plt.savefig(r"./forecast/Error-%s.png" % time_now)


plt.show()