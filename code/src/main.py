'''
Description: 
Version: 1.0
Autor: Julian Lin
Date: 2023-01-09 20:47:33
LastEditors: Julian Lin
LastEditTime: 2023-01-09 23:49:01
'''
import argparse 
import dataloader
import os

import torch.optim as optim
import torch.nn as nn
import torch
from dataloader import *
import mymodel
import math
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    # 使用tqdm来迭代训练过程并显示进度条
    with tqdm(total=num_batches) as progress_bar:
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_description(
                f"Training batch {i}/{num_batches}, loss={loss.item():.6f}")
            progress_bar.update()

    avg_loss = total_loss / num_batches
    return total_loss  # avg_loss


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return total_loss #avg_loss

def train_hyp_choose(model, train_loader, test_loader, opt):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    # 选择损失函数
    if opt.loss == 'mse':
        criterion = nn.MSELoss()
    elif opt.loss == 'mae':
        criterion = nn.L1Loss()
    elif opt.loss == 'rmse':
        criterion = math.sqrt(nn.MSELoss())
    else:
        raise ValueError('loss not found')
        return
    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    else:
        raise ValueError('optimizer not found')
        return
    for epoch in range(opt.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device=device)
        test_loss = evaluate(model, test_loader, criterion, device=device)
        print(f"Epoch {epoch}, train loss={train_loss:.6f}, test loss={test_loss:.6f}")

def main(opt):
        
    
    if opt.model == 'lstm':
        model = mymodel.LSTM(input_size=1, hidden_size=128, output_size=1, num_layers=10)
    elif opt.model == 'gru':
        model = mymodel.GRU()
    elif opt.model == 'cnn':
        model = mymodel.CNN()
    elif opt.model == 'transformer':
        model = mymodel.Transformer()
    else:
        raise ValueError('model not found')
        return

    train_data, test_data = dataloader.dataload(opt.dataset, test=0.2)
    train_loader, test_loader = data_process(train_data, test_data)
    train_hyp_choose(model, train_loader, test_loader, opt)
    


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--name', type=str, default='default')
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=1024)
    parse.add_argument('--dataset', type=str, default='../dataset/dataset_702')
    parse.add_argument('--model', type=str, default='lstm')
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--optim', type=str, default='adam')
    parse.add_argument('--loss', type=str, default='mse')
    parse.add_argument('--device', type=str, default='cpu')
    opt = parse.parse_args()
    main(opt)
