import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, features_file, labels_file, transform=None, test=False):
        self.features_pd = pd.read_csv(features_file, header=None, index_col=False)
        self.targets_pd = pd.read_csv(labels_file, header=None, index_col=False)

        combined_data = pd.concat([self.features_pd, self.targets_pd], axis=1)
        shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)
        
        self.features_pd = shuffled_data.iloc[:, :-1]
        self.targets_pd = shuffled_data.iloc[:, -1]
        self.features = self.features_pd.values
        self.targets = self.targets_pd.values.astype(int)
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features_pd.iloc[idx].values)
        label = int(self.targets_pd.iloc[idx])
        if self.transform:
            features = self.transform(features)
        return features, label

class DatasetSplit(Dataset):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.idxs = [int(i) for i in index]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def ToN_IoT():
    features_file_path_train = 'data/ToN-IoT_Data/train_data.csv'
    labels_file_path_train =  'data/ToN-IoT_Data/train_label.csv'
    features_file_path_test = 'data/ToN-IoT_Data/test_data.csv'
    labels_file_path_test =  'data/ToN-IoT_Data/test_label.csv'

    trainset = CustomDataset(features_file=features_file_path_train, labels_file=labels_file_path_train)
    testset = CustomDataset(features_file=features_file_path_test, labels_file=labels_file_path_test, test=True)
    return trainset, testset

def ton_iot_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(2021)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def ton_iot_noniid_s(dataset, num_users, skew):
    s = skew
    num_per_user = int(len(dataset)/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = dataset.targets.flatten().tolist()
    idxs = np.arange(len(dataset.targets))
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s*len(labels))
    iid_idxs = idxs_labels[0,:iid_length]
    noniid_idxs_labels = idxs_labels[:,iid_length:]
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    num_shards, num_imgs = 100, int(num_imgs_noniid/2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    np.random.seed(111)
    for i in range(num_users):
        selected_set = set(np.random.choice(all_idxs, num_imgs_iid,replace=False))
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = dict_users[i].astype(int)
        np.random.shuffle(dict_users[i])
    return dict_users

def get_ton_iot_dataset(args):
    train_dataset, test_dataset = ToN_IoT()
    if args.iid:
        user_groups = ton_iot_iid(train_dataset, args.num_users)
    else:
        user_groups = ton_iot_noniid_s(train_dataset, args.num_users, args.skew)
    
    train_loader = []
    for idx in range(args.num_users):
        loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                            batch_size=args.local_bs, shuffle=True)
        train_loader.append(loader)
    test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=True)
    return train_loader, test_loader
