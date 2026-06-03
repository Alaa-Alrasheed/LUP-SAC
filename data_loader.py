import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
#from femnist import FEMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features_file, labels_file, transform=None, test=False):
        self.features_pd = pd.read_csv(features_file,header=None, index_col=False)
        self.targets_pd = pd.read_csv(labels_file,header=None, index_col=False)

        combined_data = pd.concat([self.features_pd, self.targets_pd], axis=1)
        # Shuffle the combined data
        shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)
        # Split back into features and targets DataFrames
        self.features_pd = shuffled_data.iloc[:, :-1]
        self.targets_pd = shuffled_data.iloc[:, -1]
        self.features = self.features_pd.values
        self.targets= self.targets_pd.values.astype(int)
        self.transform = transform
        self.num_classes = 10
        self.test =  test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features =  torch.FloatTensor(self.features_pd.iloc[idx].values)
        label = int(self.targets_pd.iloc[idx])  # Assuming single label value, not array

            # Convert label to one-hot encoding
        label_one_hot = label
       
        #if self.test == False:
            #label_one_hot = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()


        
        if self.transform:
            features = self.transform(features)
        return features, label_one_hot



train_dataset, test_dataset  = AG_NEWS()
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

def build_vocabulary(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

vocab = build_vocab_from_iterator(build_vocabulary([train_dataset, test_dataset]), min_freq=50, specials=["<UNK>"])

vocab.set_default_index(vocab["<UNK>"])

#print(len(vocab))

from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

#train_dataset, test_dataset  = AG_NEWS()
#train_dataset, test_dataset  = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)
#print(train_dataset)
target_classes = ["World", "Sports", "Business", "Sci/Tech"]

max_words = 25

def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X] ## Bringing all samples to max_words length.

    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]


def AGNews_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(2021)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def AGNews_noniid_s(dataset, num_users,skew):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs
    train_loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=vectorize_batch, shuffle=True)
    all_labels = []

    for batch in train_loader:
        _, labels = batch  # Assuming the labels are in the second position of each batch
        all_labels.extend(labels.tolist())

    s = skew
    num_per_user = int(len(dataset)/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array(all_labels)
    idxs = np.arange(len(all_labels))
    # iid labels
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*len(labels))
    iid_idxs = idxs_labels[0,:iid_length]
    # noniid labels
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = 100, int(num_imgs_noniid/2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    np.random.seed(111)
    for i in range(num_users):
        # allocate iid idxs
        selected_set = set(np.random.choice(all_idxs, num_imgs_iid,replace=False))
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        # allocate noniid idxs
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        np.random.shuffle(dict_users[i])

    return dict_users


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(2021)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_class = 5
    num_imgs = int(60000/(num_class*num_users))
    num_per_label = 6000
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = dataset.targets.numpy()
    idxs = np.arange(len(labels))
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(10)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(10)]

    # divide and assign 2 shards/client
    np.random.seed(2021)
    for i in range(num_users):
        rand_label = np.random.choice(label_list, num_class, replace=False)
        for y in rand_label:
            start = y*num_per_label+label_used[y]
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+num_imgs]), axis=0)
            label_used[y] = label_used[y] + num_imgs
            if label_used[y] == num_per_label:
                label_list.remove(y)
    return dict_users



def ton_iot_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(2021)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def ton_iot_noniid_s(dataset, num_users,skew):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs
    s = skew
    num_per_user = int(len(dataset)/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = dataset.targets.flatten().tolist()
  
    idxs = np.arange(len(dataset.targets))
    # iid labels

    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*len(labels))
    iid_idxs = idxs_labels[0,:iid_length]
    # noniid labels
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = 100, int(num_imgs_noniid/2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    np.random.seed(111)
    for i in range(num_users):
        # allocate iid idxs
        selected_set = set(np.random.choice(all_idxs, num_imgs_iid,replace=False))
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        # allocate noniid idxs
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        np.random.shuffle(dict_users[i])
    return dict_users


def mnist_noniid_s(dataset, num_users,skew):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs
    s = skew
    num_per_user = int(60000/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = dataset.targets.numpy()
    idxs = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*len(labels))
    iid_idxs = idxs_labels[0,:iid_length]
    # noniid labels
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = 100, int(num_imgs_noniid/2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    np.random.seed(111)
    for i in range(num_users):
        # allocate iid idxs
        selected_set = set(np.random.choice(all_idxs, num_imgs_iid,replace=False))
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        # allocate noniid idxs
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        np.random.shuffle(dict_users[i])
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    np.random.seed(23)
    for i in range(num_users):

        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 50, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    np.random.seed(24)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
    return dict_users


def cifar_noniid_s(dataset, num_users,skew):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 50,000 training imgs
    s = skew
    num_per_user = int(50000/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*len(labels))
    iid_idxs = idxs_labels[0,:iid_length]
    # noniid labels
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = 100, int(num_imgs_noniid/2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    for i in range(num_users):
        np.random.seed(111)
        # allocate iid idxs
        selected_set = set(np.random.choice(all_idxs, num_imgs_iid,replace=False))
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        # allocate noniid idxs
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        np.random.shuffle(dict_users[i])
    return dict_users

def AGNews():
    train_dataset, test_dataset  = AG_NEWS()
    train_dataset, test_dataset  = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)
    print("AG News Data Loading...")
    return train_dataset, test_dataset


def ToN_IoT():

    features_file_path_train = 'data/ToN-IoT_Data/train_data.csv'
    labels_file_path_train =  'data/ToN-IoT_Data/train_label.csv'

    features_file_path_test = 'data/ToN-IoT_Data/test_data.csv'
    labels_file_path_test =  'data/ToN-IoT_Data/test_label.csv'

    # Create a custom dataset instance
    trainset = CustomDataset(features_file=features_file_path_train, labels_file=labels_file_path_train)
    testset = CustomDataset(features_file=features_file_path_test, labels_file=labels_file_path_test,test=True)
    return trainset, testset

def mnist():
    trainset = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    testset = datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    print("MNIST Data Loading...")
    return trainset, testset


def fmnist():
    trainset = datasets.FashionMNIST('data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    testset = datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    print("FashionMNIST Data Loading...")
    return trainset, testset


def cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)  #
    testset = datasets.CIFAR10(

        root='data', train=False, download=True, transform=transform_test)
    print("CIFAR10 Data Loading...")
    return trainset, testset



class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.idxs = [int(i) for i in index]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dataset = []
    test_dataset = []
    user_groups = {}
    train_loader = []
    test_loader = []
    if args.dataset == 'ton_iot':
        train_dataset, test_dataset = ToN_IoT()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = ton_iot_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            #if args.unbalance:
                # Chose uneuqal splits for every user
              #  raise NotImplementedError()
           # else:
                # Chose euqal splits for every user
            user_groups = ton_iot_noniid_s(train_dataset, args.num_users,args.skew)
    
    elif args.dataset == 'cifar':
        train_dataset, test_dataset = cifar10()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            #if args.unbalance:
                # Chose uneuqal splits for every user
              #  raise NotImplementedError()
           # else:
                # Chose euqal splits for every user
            user_groups = cifar_noniid_s(train_dataset, args.num_users,args.skew)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
       
        if args.dataset == 'mnist':
            train_dataset, test_dataset = mnist()
        if args.dataset == 'fmnist':
            train_dataset, test_dataset = fmnist()
        # sample training data amongst users
        if args.iid and args.dataset == 'mnist' or args.dataset == 'fmnist':
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        elif args.dataset == 'mnist' or args.dataset == 'fmnist':
            # Sample Non-IID user data from Mnist
            #if args.unbalance:
                # Chose uneuqal splits for every user
             #   raise NotImplementedError()
           # else:
                # Chose euqal splits for every user
            user_groups = mnist_noniid_s(train_dataset, args.num_users,args.skew)
    
    elif args.dataset == 'AGNews':
        train_dataset, test_dataset = AGNews()
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = AGNews_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            #if args.unbalance:
                # Chose uneuqal splits for every user
             #   raise NotImplementedError()
           # else:
                # Chose euqal splits for every user
            user_groups = AGNews_noniid_s(train_dataset, args.num_users,args.skew)

    else:
        raise NotImplementedError()
    
    if args.dataset == 'AGNews':
        for idx in range(args.num_users):
            loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), batch_size=args.local_bs, collate_fn=vectorize_batch, shuffle=True)
            train_loader.append(loader)

        test_loader  = DataLoader(test_dataset , batch_size=args.local_bs, collate_fn=vectorize_batch)

    else:
        print("here")

        for idx in range(args.num_users):
            loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                batch_size=args.local_bs, shuffle=True)
            train_loader.append(loader)

        test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=True)
    return train_loader, test_loader


if __name__ =='__main__':
    trainset, testset = AG_NEWS()
    print(len(trainset))
    print(trainset)
    print(testset.targets)