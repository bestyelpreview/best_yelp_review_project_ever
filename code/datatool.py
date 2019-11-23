import os
from utils import download_url, data_parse, preprocess_data

class YELP(object):
    """
    `yelp-review-polarity <https://www.kaggle.com/irustandi/yelp-review-polarity/version/1>`_ Dataset.
    """
    base_folder = 'yelp_review_polarity_csv'
    url = "https://storage.googleapis.com/kaggle-data-sets/169237/384451/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1574631896&Signature=SOhiNoHMEifPei4E6iTXxfrRbaYyodnmTEPbJ04tf0rP7XlRKY3hB64xsRiFSINswSfV3llTFUqzQ%2FPYqnyniUg%2F16BBpCEPh%2FK7aGUFehCD4j0rjhSR2zA5Hxx9nxnn4Cz6n%2BZxD93q5pGkZDuR5NO1sYgBlwgpdl9haDuarX6Ytbj%2Fuc4d%2BSytCnAHR3J6OxRHOGhz521Q0yESu0lEVOf2mpYIQZyDInoa5M%2F6yEYnS6YsRTKL8pvYAQkdAFBSnIzkLcpzOzYvCqtCA%2BWMpRnzqrvj3OBX%2FTLGCBDhKHIin7IQAMph%2FJSO5K6rhcI9gvLA4g4du2UT7wa1qA%2BvbA%3D%3D&response-content-disposition=attachment%3B+filename%3Dyelp-review-polarity.zip"
    filename = "yelp-review-polarity.zip"
    tgz_md5 = 'e6dfe992364fc80ec098e48edd5afc9b'

    def __init__(self, root, preprocess_path, train=True, download=False):
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
        return label, [int(x) for x in review.split()]

    def __len__(self):
        return len(self.data_set)

    def download(self):
        import zipfile
        
        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
            zip_ref.extractall(self.root)

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
