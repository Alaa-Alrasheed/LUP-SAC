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

def cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    print("CIFAR-10 Data Loading...")
    return trainset, testset

def cifar10_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(2021)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_noniid_s(dataset, num_users, skew):
    s = skew
    num_per_user = int(50000/num_users)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    
    # In torchvision.datasets.CIFAR10, targets is a python list, so we convert it to numpy array
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    iid_length = int(s * len(labels))
    iid_idxs = idxs_labels[0, :iid_length]
    noniid_idxs_labels = idxs_labels[:, iid_length:]
    
    # Sort non-iid portion by label
    idxs_noniid = noniid_idxs_labels[:, noniid_idxs_labels[1, :].argsort()]
    noniid_idxs = idxs_noniid[0, :]
    
    # Create shards. CIFAR10 has 50k images. Non-IID imgs total = 50000 * (1-s).
    # Assuming s=0.5, non-iid = 25000. num_shards = 100, num_imgs per shard = 250.
    num_shards, num_imgs = 100, int(num_imgs_noniid / 2)
    idx_shard = [i for i in range(num_shards)]
    all_idxs = [int(i) for i in iid_idxs]
    
    np.random.seed(111)
    for i in range(num_users):
        # 1) Select IID portion
        selected_set = set(np.random.choice(all_idxs, num_imgs_iid, replace=False))
        all_idxs = list(set(all_idxs) - selected_set)
        dict_users[i] = np.concatenate((dict_users[i], np.array(list(selected_set))), axis=0)
        
        # 2) Select 2 Shards for Non-IID portion
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], noniid_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            
        dict_users[i] = dict_users[i].astype(int)
        np.random.shuffle(dict_users[i])
        
    return dict_users

def get_cifar10_dataset(args):
    train_dataset, test_dataset = cifar10()
    if args.iid:
        user_groups = cifar10_iid(train_dataset, args.num_users)
    else:
        user_groups = cifar10_noniid_s(train_dataset, args.num_users, args.skew)
    
    train_loader = []
    for idx in range(args.num_users):
        loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                            batch_size=args.local_bs, shuffle=True)
        train_loader.append(loader)
    test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    return train_loader, test_loader
