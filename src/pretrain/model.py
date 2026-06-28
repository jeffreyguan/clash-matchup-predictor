import torch
import torch.nn as nn

class DeckTransformer(nn.Module):
    def __init__(self, num_cards=122, embedding_dim=64, nhead=4, dim_feedforward=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_cards, embedding_dim=embedding_dim)
        layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
        self.mlm_head=nn.Linear(in_features=embedding_dim, out_features=num_cards)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.mlm_head(x)
