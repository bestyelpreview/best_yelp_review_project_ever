#!/usr/bin/python3

from datatool import YELP
import torch
import argparse

if __name__ == "__main__":
    # set hyperparameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='the size of each batch')

    # set training parameters
    parser.add_argument('--gpu', type=bool, default=False, 
                        help='use gpu for training')
    parser.add_argument('--load', type=bool, default=False, 
                        help='load model from check point')
    parser.add_argument('--optim', type=str, default='adam', 
                        help='optimizer type, adam or rmsprop')

    # set path of data and ckp
    parser.add_argument('--data', type=str, default="./data",
                        help='path of data')
    parser.add_argument('--ckp', type=str, default="./check_point",
                        help='path of check point')

    args = parser.parse_args()


    # Load dataset with transform, download if empty 
    train_set = YELP(root=args.data, train=True, download=True)
    test_set = YELP(root=args.data, train=False, download=False)

    # Load batch data automatically
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
