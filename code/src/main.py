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
def main(opt):
        
    dataloader.dataloader(opt.dataset, test=0.2)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--name', type=str, default='default')
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--dataset', type=str, default='../dataset/dataset_702')
    parse.add_argument('--model', type=str, default='lstm')
    opt = parse.parse_args()
    main(opt)
