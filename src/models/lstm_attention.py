import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output : (batch, seq_len, hidden_size)
        scores = self.attention(lstm_output)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)        # (batch, seq_len, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, weights


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(BiLSTMAttention, self).__init__()
        
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # x2 car bidirectionnel
        self.attention = AttentionLayer(hidden_size * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x : (batch, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        
        # BatchNorm sur les features
        x = x.view(-1, input_size)
        x = self.batch_norm(x)
        x = x.view(batch_size, seq_len, input_size)
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(x)   # (batch, seq_len, hidden*2)
        
        # Attention
        context, weights = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Classification
        output = self.classifier(context)  # (batch, 1)
        
        return output.squeeze(1), weights


if __name__ == "__main__":
    # Test rapide
    batch_size = 8
    seq_len = 20
    input_size = 74
    
    model = BiLSTMAttention(input_size=input_size)
    print(model)
    
    # Fausse entrée pour tester
    x = torch.randn(batch_size, seq_len, input_size)
    output, weights = model(x)
    
    print(f"\nShape entrée  : {x.shape}")
    print(f"Shape sortie  : {output.shape}")
    print(f"Shape weights : {weights.shape}")
    print(f"Output sample : {output[:3].detach()}")
