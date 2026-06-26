import torch
import torch.nn as nn

class MatchupModel(nn.Module):
    def __init__(self, num_cards, embedding_dim=8, card_features=None):
        super().__init__()
        self.embedding = nn.Embedding(num_cards, embedding_dim=embedding_dim)

        if card_features is not None:
            self.register_buffer('card_features', card_features)
            feature_dim = card_features.shape[1]
        else:
            self.card_features = None
            feature_dim = 0

        self.network = nn.Sequential(
            nn.Linear((embedding_dim + feature_dim) * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        embedded = self.embedding(x)  # [B, 16, embedding_dim]
        if self.card_features is not None:
            features = self.card_features[x]  # [B, 16, feature_dim]
            embedded = torch.cat([embedded, features], dim=2)
        blue = embedded[:, :8, :].reshape(embedded.size(0), -1)
        red = embedded[:, 8:16, :].reshape(embedded.size(0), -1)
        combined = torch.cat([blue, red], dim=1)
        return self.network(combined).squeeze(1)
    