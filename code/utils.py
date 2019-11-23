import pandas as pd
import numpy as np
import os
import errno
import hashlib
import nltk
from tqdm import tqdm
import itertools


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def data_parse(path):
    dataset = pd.read_csv(path, sep=",")
    return dataset


def preprocess_data(data_path, preprocess_data_path, train=True):
    if train:
        if os.path.exists(os.path.join(preprocess_data_path, 'yelp_id_to_word.npy')) and \
                os.path.exists(os.path.join(preprocess_data_path, 'yelp_word_to_id.npy')) and \
                os.path.exists(os.path.join(preprocess_data_path, 'yelp_train.csv')):
            return

        raw_data_path = os.path.join(data_path, "train.csv")
        x_train, y_train = [], []
        dataset = pd.read_csv(raw_data_path, sep=",")
        t = tqdm(range(dataset.shape[0]))
        for i in t:
            t.set_description("Process train set  %s" % i)
            label, review = dataset.iloc[i, :]
            review = nltk.word_tokenize(review)
            review = [w.lower() for w in review]
            x_train.append(review)
            y_train.append(label)

        all_tokens = itertools.chain.from_iterable(x_train)
        word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

        all_tokens = itertools.chain.from_iterable(x_train)
        id_to_word = [token for idx, token in enumerate(set(all_tokens))]

        x_train_token_ids = [
            [word_to_id.get(token, -1)+1 for token in x] for x in x_train]

        np.save(os.path.join(preprocess_data_path,
                             'yelp_id_to_word.npy'), np.asarray(id_to_word))
        np.save(os.path.join(preprocess_data_path,
                             'yelp_word_to_id.npy'), np.asarray(word_to_id))

        # save training data to single text file
        with open(os.path.join(preprocess_data_path, 'yelp_train.csv'), 'w', encoding='utf-8') as fout:
            for i, tokens in enumerate(x_train_token_ids):
                fout.write(str(y_train[i])+",")
                for token in tokens:
                    fout.write("%i " % token)
                fout.write("\n")
    else:
        if os.path.exists(os.path.join(preprocess_data_path, 'yelp_test.csv')):
            return

        if not os.path.exists(os.path.join(preprocess_data_path, 'yelp_word_to_id.npy')):
            raise FileNotFoundError(
                'Dictionary not found, please process training set first')

        word_to_id = np.load(os.path.join(
            preprocess_data_path, 'yelp_word_to_id.npy')).item()

        raw_data_path = os.path.join(data_path, "test.csv")
        x_test, y_test = [], []
        dataset = pd.read_csv(raw_data_path, sep=",")
        t = tqdm(range(dataset.shape[0]))
        for i in t:
            t.set_description("Process test set  %s" % i)
            label, review = dataset.iloc[i, :]
            review = nltk.word_tokenize(review)
            review = [w.lower() for w in review]
            x_test.append(review)
            y_test.append(label)

        x_test_token_ids = [
            [word_to_id.get(token, -1)+1 for token in x] for x in x_test]

        # save test data to single text file
        with open(os.path.join(preprocess_data_path, 'yelp_test.csv'), 'w', encoding='utf-8') as fout:
            for i, tokens in enumerate(x_test_token_ids):
                fout.write(str(y_test[i])+",")
                for token in tokens:
                    fout.write("%i " % token)
                fout.write("\n")
