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

def dataloader(path, test=0.2):
    if os.path.exists(path+'.pkl'):
        data = pd.read_pickle(path + '.pkl')
    else:
        files = os.listdir(path)
        data = pd.DataFrame()
        for file in tqdm(files, desc='load data: '+path.split('/')[-1]):
            file_data = pd.read_table(os.path.join(path, file), sep=' ',header=None, dtype=str)
            file_data.iloc[:,0] = file[:-4] + file_data.iloc[:,0]
            data = pd.concat([data, file_data.iloc[:,[0,1]]], axis=0)
        
        data.columns = ['time', 'power']
        data = data.sort_values(by='time')
        data = data.drop_duplicates()
        data = data[['time','power']].apply(pd.to_numeric)
        data = data.drop(data[data.power > 10000].index)
        data.to_pickle(path + '.pkl')
        print('save data to ' + path + '.pkl')
    draw(data)
    return data

def draw(data):
    N =len(data['time'].values)
    # data['time'] = data['time'] - data['time'].values[0]
    # data['time'] = data['time'] / 1e8
    plt.plot(data['time'].values ,data['power'].values)
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