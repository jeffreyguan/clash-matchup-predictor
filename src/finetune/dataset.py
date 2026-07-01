import torch
from torch.utils.data import Dataset

class FinetuneDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(
            [item[:8] + item[8:16] for item in data],
            dtype=torch.long
        )
        self.labels = torch.tensor([item[-1] for item in data], dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    