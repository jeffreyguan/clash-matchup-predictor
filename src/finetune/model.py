import torch
import torch.nn as nn
from pretrain.model import DeckTransformer

class MatchupPredictor(nn.Module):
    def __init__(self, pretrain_path=None, num_cards=122, embedding_dim=64,
                 nhead=4, dim_feedforward=256, num_layers=2, ):
        super().__init__()
        encoder = DeckTransformer(num_cards=num_cards, embedding_dim=embedding_dim,
                                  nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers)
        if pretrain_path is not None:
            ckpt = torch.load(pretrain_path, map_location="cpu")
            encoder.load_state_dict(ckpt["model_state_dict"])
        self.embedding = encoder.embedding
        self.transformer = encoder.transformer

        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def encode(self, deck):
        x = self.embedding(deck)
        x = self.transformer(x)
        return x.mean(dim=1)

    def forward(self, x):                 
        team = self.encode(x[:, :8])
        opp  = self.encode(x[:, 8:])
        combined = torch.cat([team, opp], dim=1) 
        return self.network(combined).squeeze(-1)    
    