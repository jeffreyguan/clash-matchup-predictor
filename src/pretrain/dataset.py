import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data, n_mask=1, mask_idx=0, ignore_idx = -100, numcards=122, train=True):
        self.decks = torch.cat([
            torch.tensor([item[:8] for item in data], dtype=torch.long),    # team decks  -> [N, 8]
            torch.tensor([item[8:16] for item in data], dtype=torch.long),  # opp decks   -> [N, 8]
        ], dim=0)                                                           # stacked     -> [2N, 8]
        self.n_mask = n_mask
        self.mask_idx = mask_idx
        self.ignore_idx = ignore_idx
        self.numcards = numcards
        self.train = train

        if not self.train:
            g = torch.Generator().manual_seed(0)         
            pairs = [self._maskdeck(deck, generator=g) for deck in self.decks]
            self.val_inputs = torch.stack([p[0] for p in pairs])   # [V, 8]
            self.val_labels = torch.stack([p[1] for p in pairs])   # [V, 8]

    def __len__(self):
        return len(self.decks)
    
    def __getitem__(self, idx):
        if self.train:
            return self._maskdeck(self.decks[idx])    
        return (self.val_inputs[idx], self.val_labels[idx])                


    def _maskdeck(self, deck, generator=None):
        input_ids = deck.clone()
        labels = torch.full_like(deck, self.ignore_idx)
        idx = torch.randperm(8, generator=generator)[:self.n_mask]
        labels[idx] = deck[idx]
        
        for i in idx:
            r = torch.rand(1, generator=generator).item()                                 # bert 80/10/10 split
            if r < 0.8:
                input_ids[i] = self.mask_idx                                              # 80% mask
            elif r < 0.9:
                input_ids[i] = torch.randint(1, self.numcards, (1,), generator=generator) # 10% random card
            else:
                input_ids[i] = deck[i]                                                    # 10% same card                               

        return (input_ids, labels)
