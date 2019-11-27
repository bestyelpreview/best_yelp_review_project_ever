#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : argtools.py
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 26.11.2019
# Last Modified Date: 26.11.2019
# Last Modified By  : Zhepei Wang <zhepeiw03@gmail.com>

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cvd', type=str, default='0',
                        help='Cuda visible devices')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='the size of each batch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for loading dataset')
    # set path of data and ckp
    parser.add_argument("--preprocess_path", type=str, default="../data/preprocess_data",
                        help='path of preprocess data')
    parser.add_argument('--data_path', type=str, default="../data",
                        help='path of data')
    parser.add_argument('--ckp', type=str, default="../check_point",
                        help='path of check point')
    parser.add_argument('--out_dir', type=str, default="../out",
                        help='output directory')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random number generator')

    args = parser.parse_args()
    return args
