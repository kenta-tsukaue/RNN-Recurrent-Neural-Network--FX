# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import warnings
warnings.simplefilter('ignore')
from sklearn.preprocessing import MinMaxScaler
import datetime
from datetime import timedelta
import calendar

# RNNクラス
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        y_rnn, h = self.rnn(x, None)
        y = self.fc(y_rnn[:, -1, :])
        return y

# EarlyStoppingクラス
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# csvファイルからデータ取得
def get_Data(start_year, end_year):
    day_list = np.empty(0)
    Close = np.empty(0)
    for y in range(start_year, end_year+1):
        for m in range(1, 13):
            # 2022年は6月までしかデータが無い
            if y == 2022:
                if m >= 7:
                    break
            path = "/public/sakai/fx/data/USDJPY_" + str(y) + "/USDJPY_" + str(y) + "_" + \
                datetime.datetime(y, m, 1).strftime("%m") + ".csv"
            df = pd.read_csv(path, header=None)
            for d in range(1, calendar.monthrange(y, m)[1]+1):
                day = datetime.datetime(y, m, d).strftime("%Y.%m.%d")
                if day in df[0].unique():
                    day_list = np.append(day_list, day)
                    Close = np.append(Close, df[df[0] == day][5].values[-1])
    new_df = pd.DataFrame({"Datetime": day_list,
                           "Close": Close})
    return new_df

if __name__ == "__main__":
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 終値を抽出し次元を調整
    df = get_Data(2015, 2022)
    target = df['Close'].values.copy()
    target = target.reshape(-1, 1)
    
    # ニューラルネットワークの入力用に値域を[0, 1]に
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(target)
    target_scaled = scaler.transform(target)
    
    # スケール語の終値を画像出力
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(target_scaled)
    plt.xlabel('Time')
    plt.ylabel('Yen')
    plt.title("Closed Yen per Day")
    plt.savefig("/public/hasegawa/study_mtg/fx_prediction/image/scaled closed.png")
    
    # 訓練データとテストデータに分割
    train_size = int(len(target_scaled) * 0.8)
    test_size = len(target_scaled) - train_size
    train = target_scaled[0:train_size, :]
    test = target_scaled[train_size:, :]
    
    # 一度の予測に入力として扱うデータ量
    time_step = 30
    n_sample = train_size - time_step - 1
    # 入力データを格納する配列
    input_data = np.zeros((n_sample, time_step, 1))
    # 正解データを格納する配列
    correct_input_data = np.zeros((n_sample, 1))
    for i in range(n_sample):
        input_data[i] = target_scaled[i: i+time_step].reshape(-1, 1)
        correct_input_data[i] = target_scaled[i+time_step: i+time_step+1]
    
    # ニューラルネットで扱えるようデータを調整
    input_data = torch.tensor(input_data, dtype=torch.float)
    correct_input_data = torch.tensor(correct_input_data, dtype=torch.float)
    dataset = torch.utils.data.TensorDataset(input_data, correct_input_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # モデルの定義
    n_inputs = 1
    n_outputs = 1
    n_hidden = 64
    n_layers = 1
    model = RNN(n_inputs, n_outputs, n_hidden, n_layers)
    batch_size = 32
    summary(model, (batch_size, time_step, 1))
    early_stopping = EarlyStopping(patience=10)
    
    # 損失関数
    loss_fnc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_record = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 200
    
    model.to(device)
    for i in range(epochs+1):
        # 学習モード
        model.train()
        # 記録用lossを初期化
        running_loss = 0.0
        for j, (x, t) in enumerate(train_loader):
            x = x.to(device)
            # 勾配初期化
            optimizer.zero_grad()
            y = model(x)
            y = y.to('cpu')
            # 勾配計算
            loss = loss_fnc(y, t)
            # 誤差逆伝播
            loss.backward()
            # 勾配更新
            optimizer.step()
            running_loss += loss.item()
        running_loss /= j+1
        loss_record.append(running_loss)
        early_stopping(running_loss)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break
        
        # 学習過程の可視化
        if i % 20 == 0:
            print('Epoch: ', i, '  Loss_train: ', running_loss)
            input_train = list(input_data[0].reshape(-1))
            predicted_train_plot = []
            # 評価モード
            model.eval()
            for k in range(n_sample):
                x = torch.tensor(input_train[-time_step:])
                x = x.reshape(1, time_step, 1)
                x = x.to(device).float()
                y = model(x)
                y = y.to('cpu')
                
                if k <= n_sample - 2:
                    input_train.append(input_data[k+1][time_step-1].item())
                predicted_train_plot.append(y[0].item())
                
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(range(len(target_scaled)), target_scaled, label='correct')
            ax.plot(range(time_step, time_step+len(predicted_train_plot)), predicted_train_plot, label='predicted')
            plt.legend()
            plt.savefig("/public/hasegawa/study_mtg/fx_prediction/image/fx predicted by RNN epochs "+str(i)+".png")
    
    # 誤差を可視化
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(len(loss_record)), loss_record, label='train')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("/public/hasegawa/study_mtg/fx_prediction/image/loss of predicted fx by RNN.png")
    
    # テストデータの学習
    test_data = np.zeros((test_size+time_step, time_step, 1))
    correct_test_data = np.zeros((test_size, 1))
    
    start_test = train_size - time_step
    for i in range(test_size):
        test_data[i] = target_scaled[start_test+i: start_test+i+time_step].reshape(-1, 1)
        correct_test_data[i] = target_scaled[start_test+i+time_step: start_test+i+time_step+1]
    input_test = list(test_data[0].reshape(-1))
    
    predicted_test_plot = []
    model.eval()
    for k in range(test_size):
        x = torch.tensor(input_test[-time_step:])
        x = x.reshape(1, time_step, 1)
        x = x.to(device).float()
        y = model(x)
        y = y.to('cpu')
        if k <= test_size - 2:
            input_test.append(test_data[k+1][time_step-1].item())
        predicted_test_plot.append(y[0].item())
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(len(target_scaled)), target_scaled, label='correct')
    ax.plot(range(start_test+time_step, start_test+time_step+len(predicted_test_plot)), predicted_test_plot, label='predicted')
    plt.legend()
    plt.savefig("/public/hasegawa/study_mtg/fx_prediction/image/test of predicted fx by RNN.png")
    
    # 予測結果をデータフレームに格納する
    train_df_day = df['Datetime'].values[time_step: len(predicted_train_plot)+time_step]
    train_df_close = df['Close'].values[time_step: len(predicted_train_plot)+time_step]
    train_df_pred = scaler.inverse_transform(np.array(predicted_train_plot).reshape(-1, 1)).reshape(-1)
    test_df_day = df['Datetime'].values[-len(predicted_test_plot):]
    test_df_close = df['Close'].values[-len(predicted_test_plot):]
    test_df_pred = scaler.inverse_transform(np.array(predicted_test_plot).reshape(-1, 1)).reshape(-1)
    train_df = pd.DataFrame({'Datetime': train_df_day,
                             'Close': train_df_close,
                             'Pred': train_df_pred})
    test_df = pd.DataFrame({'Datetime': test_df_day,
                            'Close': test_df_close,
                            'Pred': test_df_pred})
    
    train_df.to_csv("/public/hasegawa/study_mtg/fx_prediction/result of train predicted fx.csv")
    test_df.to_csv("/public/hasegawa/study_mtg/fx_prediction/result of test predicted fx.csv")
    
    # 平均二乗誤差（MSE）を算出
    from sklearn.metrics import mean_squared_error as MSE
    print("Train MSE values: {:.3f}".format(MSE(train_df_close, train_df_pred)))
    print("Test MSE values: {:.3f}".format(MSE(test_df_close, test_df_pred)))
