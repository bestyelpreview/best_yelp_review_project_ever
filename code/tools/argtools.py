#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : argtools.py
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 26.11.2019
# Last Modified Date: 26.11.2019
# Last Modified By  : Zhepei Wang <zhepeiw03@gmail.com>

"""!
@brief Pytorch argument parser for the experiments.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse


def create_model_parameters(parser, create_list_model_args=False):
    # Model Configuration Parameters
    if create_list_model_args:
        parser.add_argument('--vocab_size', type=int, default=[8000],
                            nargs='+', help='vocabulary size')
        parser.add_argument("-M", "--model_type", type=str, nargs='+',
                            help="The type of model you would like to use for "
                                 "prediction. LSTM? A bidirectional LSTM "
                                 "(BLSTM) or a BLSTM with Attention on "
                                 "top (BLSTM-Att)",
                            default=['LSTM', 'BLSTM', 'BLSTM-Att'],
                            choices=['LSTM', 'BLSTM', 'BLSTM-Att'])
        parser.add_argument("-L", "--num_layers", type=int, nargs='+',
                            help="Number of hidden layers in the RNN.",
                            default=[1, 2, 3])
        parser.add_argument("-H", "--num_hidden_units", type=int, nargs='+',
                            help="Number of hidden units for each layer in "
                                 "the selected model.",
                            default=[128, 256])
        parser.add_argument("-D", "--dropout_rate", type=float, nargs='+',
                            help="Dropout Rate applied on all layers.",
                            default=[0.0, 0.3])
        parser.add_argument("-E", "--embedding_size", type=int, nargs='+',
                            help="The size of the output of the embedding "
                                 "layer for each token .",
                            default=[256, 512])
    else:
        parser.add_argument('--vocab_size', type=int, default=8000,
                            help='vocabulary size')
        parser.add_argument("-M", "--model_type", type=str,
                            help="The type of model you would like to use for "
                                 "prediction. LSTM? A bidirectional LSTM "
                                 "(BLSTM) or a BLSTM with Attention on "
                                 "top (BLSTM-Att)",
                            default='LSTM',
                            choices=['LSTM', 'BLSTM', 'BLSTM-Att'])
        parser.add_argument("-L", "--num_layers", type=int,
                            help="Number of hidden layers in the RNN.",
                            default=2)
        parser.add_argument("-H", "--num_hidden_units", type=int,
                            help="Number of hidden units for each layer in "
                                 "the selected model.",
                            default=512)
        parser.add_argument("-D", "--dropout_rate", type=float,
                            help="Dropout Rate applied on all layers.",
                            default=0.1)
        parser.add_argument("-E", "--embedding_size", type=int,
                            help="The size of the output of the embedding "
                                 "layer for each token .",
                            default=64)


def get_args(parallel_experiments=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                            available for running this experiment""",
                        default=['2'],
                        choices=['0', '1', '2', '3'])
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int,
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
    parser.add_argument('--wandb_project', type=str, default='cs547',
                        help='wandb project name')
    parser.add_argument('--wandb_entity', type=str,
                        help='wandb entity')

    create_model_parameters(parser,
                            create_list_model_args=parallel_experiments)

    args = parser.parse_args()
    return args
