#!/usr/bin/python3
"""!
@brief Experiment runnign using pytorch LSTM predictor wrapper

@authors Zhepei Wang <zhepeiw03@gmail.com>
         Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

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
from models import LSTMPredictorWrapper
from tools.misc import reset_seed

import pdb

def trainer(args, model, train_loader, val_loader,
            out_dir, lr=0.01, max_epochs=45):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.BCEWithLogitsLoss()

    start_ep = 0
    out_cache = []
    cache_limit = 2
    all_res = []
    for ep in range(start_ep, max_epochs):
        # train
        print('==> Start training epoch {} / {}'.format(ep, max_epochs))
        t_start = time.time()
        model = model.cuda()
        model.train()
        curr_train_loss = []
        for i, (x, y, lengths) in enumerate(tqdm(train_loader)):
            x = x.cuda()
            y = y.cuda()
            lengths = lengths.cuda()
            y_pred = model(x, lengths)
            loss = loss_fn(y_pred.squeeze(), y.float())
            if np.isnan(loss.item()):
                pdb.set_trace()
            curr_train_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(np.array(curr_train_loss).mean())
            break

        print('====> Training loss = {:.6f}'.format(np.array(curr_train_loss).mean()))
        # validate
        model.eval()
        curr_val_correct = 0
        curr_val_total = 0
        curr_val_loss = []
        for i, (x, y, lengths) in enumerate(tqdm(val_loader)):
            x = x.cuda()
            with torch.no_grad():
                x = x.cuda()
                y = y.cuda()
                lengths = lengths.cuda()
                y_pred = model(x, lengths)
                loss = loss_fn(y_pred.squeeze(), y.float())
                if np.isnan(loss.item()):
                    pdb.set_trace()
                y_pred_bin = (torch.sigmoid(y_pred) > 0.5).long()
                curr_val_correct += sum(y_pred_bin.squeeze() == y)
                curr_val_total += len(y)
            curr_val_loss.append(loss.item())
            print(np.array(curr_val_loss).mean())
            break

        val_acc = curr_val_correct.float().item() / curr_val_total
        print('====> Validation loss = {:.6f}'.format(val_acc))
        val_stat = np.array(curr_val_loss).mean()
        print('====> Validation loss = {:.6f}'.format(val_stat))
        scheduler.step(val_stat)
        t_ep = time.time() - t_start
        print('Time elapsed: {:.3f}s'.format(t_ep))

        # dump
        curr_res = {}
        curr_res['train_loss'] = curr_train_loss
        curr_res['val_loss'] = curr_val_loss
        curr_res['val_acc'] = val_acc
        curr_res['time'] = t_ep
        all_res.append(curr_res)
        pickle.dump(all_res, open(os.path.join(out_dir, 'res/results.pk'), 'wb'))
        model = model.cpu()

        # logging to wandb
        if args.wandb_entity is not None:
            wandb.log({
                'train_loss': np.array(curr_train_loss).mean(),
                'val_loss': val_stat,
                'val_acc': val_acc
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


def experiment2string(args):
    cor_string = 'M_{}_L_{}_V_{}_E_{}_H_{}_Dr_{}'.format(
        args.model_type, args.num_layers, args.vocab_size,
        args.embedding_size, args.num_hidden_units,
        args.dropout_rate
    )
    return cor_string


def safe_run_experiment(args):
    try:
        run_experiment(args)
        raised_exception = False
    except Exception as e:
        raised_exception = e
    finally:
        torch.cuda.empty_cache()
        if raised_exception:
            raise raised_exception


def run_experiment(args):
    #  cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [cad for cad in args.cuda_available_devices])
    reset_seed(args.seed)
    # Install the punkt package
    if not os.path.exists("../nltk_data"):
        os.mkdir("../nltk_data")
        nltk.download('punkt', download_dir="../nltk_data")
    nltk.data.path.append("../nltk_data")

    # Load dataset with proprocessing, download if empty. Preprocess will only do once.
    train_set = YELP(root=args.data_path,
                     preprocess_path=args.preprocess_path,
                     train=True, download=True,
                     vocab_size=args.vocab_size)
    test_set = YELP(root=args.data_path,
                    preprocess_path=args.preprocess_path,
                    train=False, download=False,
                    vocab_size=args.vocab_size)
    # Load batch data automatically
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, collate_fn=PadSequence()
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, collate_fn=PadSequence()
    )

    #  models
    reset_seed(args.seed)
    model = LSTMPredictorWrapper(vocabulary_size=args.vocab_size + 1,
                                 embedding_dimension=args.embedding_size,
                                 dropout_rate=args.dropout_rate,
                                 num_layers=args.num_layers,
                                 num_hidden_units=args.num_hidden_units,
                                 LSTM_type=args.model_type,
                                 return_attention_weights=False)
    print("Model Params {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # output
    specific_folder = experiment2string(args)

    out_dir = os.path.join('./out/', specific_folder)
    if not os.path.exists(out_dir + '/res'):
        os.makedirs(out_dir + '/res')
    if not os.path.exists(out_dir + '/models'):
        os.makedirs(out_dir + '/models')

    # wandb
    if args.wandb_entity is not None:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name='yelp_pid{}'.format(os.getpid()))

    #  training
    trainer(args, model, train_loader, test_loader, out_dir,
            lr=args.lr, max_epochs=args.epochs)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    run_experiment(args)


