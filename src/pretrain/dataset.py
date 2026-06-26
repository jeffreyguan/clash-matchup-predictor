import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PretrainDataset(Dataset):
    def __init__(self, decks, n_mask=1, mask_idx=0, numcards=122, train=True):
        