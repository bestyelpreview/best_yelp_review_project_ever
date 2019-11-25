import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import download_url, data_parse, preprocess_data
import pdb

class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        seq_x = [x[0] for x in sorted_batch]
        seq_x_padded = torch.nn.utils.rnn.pad_sequence(seq_x, batch_first=True)
        seq_y = [x[1] for x in sorted_batch]
        seq_y = torch.LongTensor([seq_y]).squeeze()
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in seq_x])
        return seq_x_padded, seq_y, lengths

class YELP(object):
    """
    `yelp-review-polarity <https://www.kaggle.com/irustandi/yelp-review-polarity/version/1>`_ Dataset.
    """
    base_folder = 'yelp_review_polarity_csv'
    file_id = "11p4HIVr8cKM-CdcZX5zkaPHFJ9whWARi"
    filename = "yelp-review-polarity.zip"
    tgz_md5 = 'e6dfe992364fc80ec098e48edd5afc9b'

    def __init__(self, root, preprocess_path, train=True, download=False, max_len=8000):
        """
        Args:
            root (string): Root directory of dataset where directory ``yelp_review_polarity_csv``
                exists or will be saved to if download is set to True.
            preprocess_path （string): Directory of preprocess dataset, and the directory will be
                created if not exists.
            train (bool, optional): If True, creates dataset from training set,
                otherwise creates from test set.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
                If dataset is already downloaded, it is not downloaded again.
        """
        self.root = root
        self.preprocess_path = preprocess_path
        self.train = train

        if download:
            self.download()

        if not os.path.exists(self.preprocess_path):
            os.mkdir(self.preprocess_path)

        data_path = os.path.join(self.root, self.base_folder)
        preprocess_data(data_path, self.preprocess_path, self.train)

        self.data_set = self.get_data(train=self.train)

    def __getitem__(self, index):
        label, review = self.data_set.iloc[index, :]

        return torch.LongTensor([int(x) for x in review.split()]), label

    def __len__(self):
        return len(self.data_set)

    def download(self):
        download_url(self.file_id, self.root, self.filename, self.tgz_md5)

    def get_data(self, train=True):
        """
        Args:
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
        """
        self.train = train
        if self.train:
            path = os.path.join(self.preprocess_path) + '/yelp_train.csv'
        else:
            path = os.path.join(self.preprocess_path) + '/yelp_test.csv'
        dataset = data_parse(path)
        return dataset
