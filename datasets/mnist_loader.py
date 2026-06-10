import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch

class DatasetSplit(Dataset):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.idxs = [int(i) for i in index]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

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

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(2021)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid_s(dataset, num_users, skew):
    s = skew
    num_per_user = int(60000/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = dataset.targets.numpy()
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

def get_mnist_dataset(args):
    train_dataset, test_dataset = mnist()
    if args.iid:
        user_groups = mnist_iid(train_dataset, args.num_users)
    else:
        user_groups = mnist_noniid_s(train_dataset, args.num_users, args.skew)
    
    train_loader = []
    for idx in range(args.num_users):
        loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                            batch_size=args.local_bs, shuffle=True)
        train_loader.append(loader)
    test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=True)
    return train_loader, test_loader
