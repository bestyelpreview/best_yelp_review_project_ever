#!/usr/bin/python3

import torch
from torch.utils.data import DataLoader
import argparse
import os
import nltk
from data_loader.datatool import YELP, PadSequence
from tools.argtools import get_args

import pdb
if __name__ == "__main__":
    args = get_args()

    # Install the punkt package
    if not os.path.exists("../nltk_data"):
        os.mkdir("../nltk_data")
        nltk.download('punkt', download_dir="../nltk_data")
    nltk.data.path.append("../nltk_data")

    # Load dataset with proprocessing, download if empt. Preprocess will only do once.
    train_set = YELP(root=args.data_path, preprocess_path=args.preprocess_path,
                     train=True, download=True, vocab_size=8000)
    test_set = YELP(root=args.data_path, preprocess_path=args.preprocess_path,
                    train=False, download=False, vocab_size=8000)
    pdb.set_trace()

    # Load batch data automatically
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=PadSequence()
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=PadSequence()
    )
    tmp = next(iter(train_loader))

