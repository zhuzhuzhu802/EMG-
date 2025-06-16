import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut

class FusedEMGDataset(Dataset):
    def __init__(self, big_path, small_path, label_path):
        self.big_path = big_path
        self.small_path = small_path
        self.label_path = label_path

        self.big = np.load(big_path, mmap_mode='r')    # 不读取，只映射
        self.small = np.load(small_path, mmap_mode='r')
        self.label = np.load(label_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # 每次索引时才读取一条，不会把全部拷进内存
        big = torch.tensor(self.big[idx], dtype=torch.float32).unsqueeze(0)       # [1, 8, 8]
        small = torch.tensor(self.small[idx], dtype=torch.float32).unsqueeze(1)   # [31, 1, 8, 8]
        label = torch.tensor(self.label[idx], dtype=torch.long)
        return big, small, label


def file_exists(filename, folder):
    return os.path.exists(os.path.join(folder, filename))



def prepare_fused_data_loaders(data_folder, batch_size):
    subject_ids = [f"AB{i:02d}" for i in range(101, 111)]
    data_files = []

    # 只存路径，不加载数据
    for sid in subject_ids:
        big = os.path.join(data_folder, f"big_pearsons_{sid}_8channel.npy")
        small = os.path.join(data_folder, f"small_seq_pearsons_{sid}_8channel.npy")
        label = os.path.join(data_folder, f"labels_{sid}_8channel.npy")
        if os.path.exists(big) and os.path.exists(small) and os.path.exists(label):
            data_files.append((big, small, label))


    loo = LeaveOneOut()
    data_loaders = []

    for train_idx, test_idx in loo.split(data_files):
        # 构造训练集
        train_datasets = [
            FusedEMGDataset(*data_files[i]) for i in train_idx
        ]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True
        )

        # 构造测试集（只有1人）
        test_dataset = FusedEMGDataset(*data_files[test_idx[0]])
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            num_workers=0, pin_memory=True, drop_last=False
        )

        data_loaders.append((train_loader, test_loader))

    return data_loaders