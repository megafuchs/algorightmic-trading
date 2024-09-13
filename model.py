import torch.nn as nn

# LSTM with Layer Normlization 

class LSTMWithLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, output_size):
        super(LSTMWithLayerNorm, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)  # Add layer normalization
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.layer_norm(out)  # Apply LayerNorm after LSTM
        out = self.fc(out[:, -1, :])  # Only take the output of the last time step
        return out

def create_model(input_size, hidden_size, num_layers, dropout_rate, output_size):
    model = LSTMWithLayerNorm(input_size, hidden_size, num_layers, dropout_rate, output_size)
    return model
