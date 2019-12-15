#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : vis_results.py
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 10.12.2019
# Last Modified Date: 10.12.2019
# Last Modified By  : Zhepei Wang <zhepeiw03@gmail.com>
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import glob
import pdb
import numpy as np

def get_results(path):
    res_files = sorted(glob.glob(os.path.join(path, '*/*/*.pk')))
    best_acc = 0
    best_acc_idx = -1
    for idx, fp in enumerate(res_files):
        spec = fp.split('/')[-3]
        res = pickle.load(open(fp,'rb'))
        val_accs = [curr['val_acc'] for curr in res]
        if max(val_accs) > best_acc:
            best_acc = max(val_accs)
            best_acc_idx = idx
        print('Model is {} with test accuracy {:.4f}'.format(spec, max(val_accs)))
    fp = res_files[best_acc_idx]
    spec = fp.split('/')[-3]
    res = pickle.load(open(fp,'rb'))
    train_losses = [np.mean(curr['train_loss']) for curr in res]
    val_losses = [np.mean(curr['val_loss']) for curr in res]
    times = [curr['time'] for curr in res]
    val_accs = [curr['val_acc'] for curr in res]
    print('Best model is {} with test accuracy {:.4f}'.format(spec, max(val_accs)))
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='test loss')
    plt.xlabel('number of epoch')
    plt.ylabel('loss value')
    plt.title('Average Training and Test Loss')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(val_accs, label='test accuracy')
    plt.xlabel('number of epoch')
    plt.ylabel('accuracy')
    plt.title('Test accuracy')
    plt.savefig('{}_res.pdf'.format(spec))
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    path = './out'
    get_results(path)
