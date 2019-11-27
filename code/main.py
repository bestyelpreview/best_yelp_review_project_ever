#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os, time, pickle
import nltk
from data_loader.datatool import YELP, PadSequence
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from tools.argtools import get_args
from modules import LSTM_Net

import pdb

def trainer(model, train_loader, val_loader,
            device, out_dir, lr=0.01, max_epochs=45):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.BCELoss()

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
            y_pred = model(x, lengths)
            loss = loss_fn(y_pred, y)
            if np.isnan(loss.item()):
                pdb.set_trace()
            curr_train_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        print('====> Training loss = {:.6f}'.format(np.array(curr_train_loss).mean()))
        #  # validate
        #  model.eval()
        #  curr_val_loss = []
        #  for i, (x, _) in enumerate(tqdm(val_loader)):
        #      x = x.to(device)
        #      with torch.no_grad():
        #          ze, zq, x_pred = model(x)
        #          loss_recon = loss_fn(x, x_pred)
        #          loss_vq = loss_fn(ze.detach(), zq)
        #          loss_cmt = loss_fn(ze, zq.detach())
        #          loss = loss_recon + loss_vq + loss_cmt * 0.25
        #      curr_val_loss.append(loss.item())
        #  val_stat = np.array(curr_val_loss).mean()
        #  print('====> Validation loss = {:.6f}'.format(val_stat))
        #  scheduler.step(val_stat)
        #  t_ep = time.time() - t_start
        #  print('Time elapsed: {:.3f}s'.format(t_ep))
        #
        #  # dump
        #  curr_res = {}
        #  curr_res['train_loss'] = curr_train_loss
        #  curr_res['val_loss'] = curr_val_loss
        #  curr_res['time'] = t_ep
        #  all_res.append(curr_res)
        #  pickle.dump(all_res, open(os.path.join(out_dir, 'res/results.pk'), 'wb'))
        #  model = model.cpu()
        #  # visualize
        #  x_orig = make_grid(denormalize(x[:16].cpu().data)).numpy()
        #  x_recon = make_grid(denormalize(x_pred[:16].cpu().data)).numpy()
        #  plt.subplot(211)
        #  plt.imshow(np.transpose(x_orig, (1, 2, 0)), interpolation='nearest')
        #  plt.title('Data')
        #  plt.subplot(212)
        #  plt.imshow(np.transpose(x_recon, (1, 2, 0)), interpolation='nearest')
        #  plt.title('Reconstruction')
        #  plt.tight_layout()
        #  # logging
        #  wandb.log({
        #      'train_loss': np.array(curr_train_loss).mean(),
        #      'val_loss': val_stat,
        #      'visualization': wandb.Image(plt)
        #  })
        #  plt.close()
        #
        #  if len(out_cache) < cache_limit or val_stat < out_cache[-1][0]:
        #      print('**** New top model found at epoch {}'.format(ep))
        #      if len(out_cache) > 0 and val_stat < out_cache[0][0]:
        #          print('=== This is a new optimal ====')
        #      # pop item
        #      if len(out_cache) >= cache_limit:
        #          old = out_cache.pop()
        #          os.remove(old[1])
        #          print('***** Popping model {}'.format(os.path.basename(old[1])))
        #
        #      ckpt = {
        #          'model_state_dict': model.state_dict(),
        #          'opt_state_dict': opt.state_dict(),
        #          'sche_state_dict': scheduler.state_dict(),
        #          'epoch': ep
        #      }
        #      fresh = os.path.join(out_dir, 'models/ckpt_ep{}.pt'.format(ep))
        #      if len(out_cache) > 0 and val_stat < out_cache[0][0]:
        #          fresh = os.path.join(out_dir, 'models/ckpt_ep{}_best.pt'.format(ep))
        #      torch.save(ckpt, fresh)
        #      out_cache.append((val_stat, fresh))
        #      out_cache = sorted(out_cache)

if __name__ == "__main__":
    args = get_args()

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
    model = LSTM_Net(vocab_size=8000, embed_size=512, hidden_size=512)
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

