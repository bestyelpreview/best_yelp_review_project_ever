#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os, time, pickle
import nltk
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from tools.argtools import get_args
from data_loader.datatool import YELP, PadSequence
from modules import LSTM_Net
from tools.misc import reset_seed

import pdb

def trainer(model, train_loader, val_loader,
            device, out_dir, lr=0.01, max_epochs=45):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.BCEWithLogitsLoss()

    start_ep = 0
    out_cache = []
    cache_limit = 5
    all_res = []
    for ep in range(start_ep, max_epochs):
        # train
        print('==> Start training epoch {} / {}'.format(ep, max_epochs))
        t_start = time.time()
        model = model.to(device)
        model.train()
        curr_train_loss = []
        for i, (x, y, lengths) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            y_pred = model(x, lengths)
            loss = loss_fn(y_pred.squeeze(), y.float())
            if np.isnan(loss.item()):
                pdb.set_trace()
            curr_train_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        print('====> Training loss = {:.6f}'.format(np.array(curr_train_loss).mean()))
        # validate
        model.eval()
        curr_val_loss = []
        for i, (x, y, lengths) in enumerate(tqdm(val_loader)):
            x = x.to(device)
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                lengths = lengths.to(device)
                y_pred = model(x, lengths)
                loss = loss_fn(y_pred.squeeze(), y.float())
                if np.isnan(loss.item()):
                    pdb.set_trace()
            curr_val_loss.append(loss.item())
        val_stat = np.array(curr_val_loss).mean()
        print('====> Validation loss = {:.6f}'.format(val_stat))
        scheduler.step(val_stat)
        t_ep = time.time() - t_start
        print('Time elapsed: {:.3f}s'.format(t_ep))

        # dump
        curr_res = {}
        curr_res['train_loss'] = curr_train_loss
        curr_res['val_loss'] = curr_val_loss
        curr_res['time'] = t_ep
        all_res.append(curr_res)
        pickle.dump(all_res, open(os.path.join(out_dir, 'res/results.pk'), 'wb'))
        model = model.cpu()
        # logging
        wandb.log({
            'train_loss': np.array(curr_train_loss).mean(),
            'val_loss': val_stat,
        })
        plt.close()

        if len(out_cache) < cache_limit or val_stat < out_cache[-1][0]:
            print('**** New top model found at epoch {}'.format(ep))
            if len(out_cache) > 0 and val_stat < out_cache[0][0]:
                print('=== This is a new optimal ====')
            # pop item
            if len(out_cache) >= cache_limit:
                old = out_cache.pop()
                os.remove(old[1])
                print('***** Popping model {}'.format(os.path.basename(old[1])))

            ckpt = {
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'sche_state_dict': scheduler.state_dict(),
                'epoch': ep
            }
            fresh = os.path.join(out_dir, 'models/ckpt_ep{}.pt'.format(ep))
            if len(out_cache) > 0 and val_stat < out_cache[0][0]:
                fresh = os.path.join(out_dir, 'models/ckpt_ep{}_best.pt'.format(ep))
            torch.save(ckpt, fresh)
            out_cache.append((val_stat, fresh))
            out_cache = sorted(out_cache)

if __name__ == "__main__":
    #  cuda
    device = torch.device('cuda:0')
    args = get_args()
    reset_seed(args.seed)
    # Install the punkt package
    if not os.path.exists("../nltk_data"):
        os.mkdir("../nltk_data")
        nltk.download('punkt', download_dir="../nltk_data")
    nltk.data.path.append("../nltk_data")

    # Load dataset with proprocessing, download if empty. Preprocess will only do once.
    train_set = YELP(root=args.data_path, preprocess_path=args.preprocess_path,
                     train=True, download=True, vocab_size=8000)
    test_set = YELP(root=args.data_path, preprocess_path=args.preprocess_path,
                    train=False, download=False, vocab_size=8000)
    # Load batch data automatically
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=PadSequence()
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=PadSequence()
    )

    #  models
    reset_seed(args.seed)
    model = LSTM_Net(vocab_size=8000+1, embed_size=512, hidden_size=512)
    print("Model Params {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # output
    out_dir = './out/AE'
    if not os.path.exists(out_dir + '/res'):
        os.makedirs(out_dir + '/res')
    if not os.path.exists(out_dir + '/models'):
        os.makedirs(out_dir + '/models')

    # wandb
    wandb.init(project='cs547',
               entity='zhepeiw2',
               name='yelp_pid{}'.format(os.getpid()))

    #  training
    trainer(model, train_loader, test_loader, device, out_dir,
           lr=args.lr, max_epochs=args.epochs)

