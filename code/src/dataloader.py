'''
Description: 
Version: 1.0
Autor: Julian Lin
Date: 2023-01-09 20:47:42
LastEditors: Julian Lin
LastEditTime: 2023-01-09 23:49:16
'''
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler


def dataload(path, test=0.2, split_mode=None):
    if os.path.exists(path+'.pkl'):
        data = pd.read_pickle(path + '.pkl')
    else:
        files = os.listdir(path)
        data = pd.DataFrame()
        for file in tqdm(files, desc='load data: '+path.split('/')[-1]):
            file_data = pd.read_table(os.path.join(
                path, file), sep=' ', header=None, dtype=str)
            file_data.iloc[:, 0] = file[:-4] + file_data.iloc[:, 0]
            data = pd.concat([data, file_data.iloc[:, [0, 1]]], axis=0)

        data.columns = ['time', 'power']
        data = data.sort_values(by='time')
        data = data.drop_duplicates()
        data = data[['time', 'power']].apply(pd.to_numeric)
        data[data['power'] > 10000] = 0
        data.to_pickle(path + '.pkl')
        print('save data to ' + path + '.pkl')
    draw(data)
    train_ratio = 1 - test  # 训练集比例
    train_size = int(len(data) * train_ratio)

    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]
    return train_data, test_data

def data_process(train_data, test_data):
        # 创建数据集对象
    train_dataset = TimeSeriesDataset(train_data)
    test_dataset = TimeSeriesDataset(test_data)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def draw(data):
    N = len(data['time'].values)
    # data['time'] = data['time'] - data['time'].values[0]
    # data['time'] = data['time'] / 1e8
    plt.plot(data['time'].values, data['power'].values)
    # change x internal size
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    # maxsize = max([t.get_window_extent().width for t in tl])
    maxsize = 30
    m = 0.01  # inch margin
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, df, look_back=24):
        self.data = df['power'].values.reshape(-1, 1)
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, index):
        x = self.data[index:index+self.look_back]
        y = self.data[index+self.look_back]
        return torch.tensor(x).float(), torch.tensor(y).float()
