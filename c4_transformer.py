

# c4_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class TransformerAgent(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, ff_dim, n_layers, output_dim, dropout=0.1):
        super(TransformerAgent, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_heads, ff_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        print(f"forward Input x.shape: {x.shape}")
        x = self.embedding(x)  # Shape: [batch_size, embed_dim]
        print(f"forward Embedding x.shape: {x.shape}")
        x = self.pos_encoder(x.unsqueeze(1))  # Shape: [batch_size, 1, embed_dim]
        print(f"forward Positional Encoding x.shape: {x.shape}")
        x = self.transformer_encoder(x)  # Shape: [batch_size, 1, embed_dim]
        print(f"forward Transformer Encoder x.shape: {x.shape}")
        x = self.fc(x.squeeze(1))  # Shape: [batch_size, output_dim]
        print(f"forward Output x.shape: {x.shape}")
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(f"PE Input x.shape: {x.shape}")
        x = x + self.pe[:, :x.size(1), :]
        print(f"PE Output x.shape: {x.shape}")
        return self.dropout(x)