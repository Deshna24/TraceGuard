import torch
import torch.nn as nn

class SequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=3, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths):
        # x: (batch, seq_len, dim)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.dropout(hn[-1])
        return self.fc(out)

class LightweightTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        # Ensure input_dim is divisible by num_heads. 384 is divisible by 4.
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x, lengths):
        # Create mask for padding
        max_len = x.size(1)
        mask = torch.arange(max_len)[None, :] >= torch.tensor(lengths)[:, None]
        mask = mask.to(x.device)
        
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Mean pooling over valid lengths
        out_pooled = []
        for i, length in enumerate(lengths):
            out_pooled.append(out[i, :length, :].mean(dim=0))
        out_pooled = torch.stack(out_pooled)
        
        return self.fc(out_pooled)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        
import math