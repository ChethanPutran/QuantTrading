import numpy as np
import torch
import torch.nn as nn


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required for TransformerModel")

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_len, n_features, output_len, d_model=64, num_heads=8, num_layers=6, use_attention=True,model_path="transformer_stock_model.pth"):
        _require_torch()
        super(TransformerModel, self).__init__()
        
        self.input_len = input_len
        self.n_features = n_features
        self.output_len = output_len
        self.use_attention = use_attention
        self.model_path=model_path
        
        # Model layers
        self.embedding = nn.Linear(n_features, d_model)  # Embedding layer to convert input into d_model dimension
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_len, d_model))  # Positional Encoding
        
        # Transformer Encoder and Decoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        
        # # Attention layer if specified
        # if use_attention:
        #     self.attention = Attention(d_model)
        
        # Output fully connected layer
        self.fc = nn.Linear(d_model, output_len)

    def forward(self, x):
        # x: [batch_size, seq_len, n_features]
        x = self.embedding(x) + self.positional_encoding  # Add positional encoding
        
        # Transformer Encoder
        encoder_output = self.encoder(x)
        
        # If attention is used, apply it after the encoder
        # if self.use_attention:
        #     encoder_output, attention_weights = self.attention(encoder_output)
        
        # For the decoder, we use the encoder output and target sequence (e.g., for seq2seq tasks)
        # Here we assume the target sequence is the same as the input sequence (for simplicity)
        decoder_output = self.decoder(encoder_output, encoder_output)
        
        # Get the output from the last time step
        # decoder_output = decoder_output[-1, :, :]
        
        # Output layer
        out = self.fc(decoder_output[:, -1, :])
        return out
        
    def save_model(self):
        _require_torch()
        torch.save(self.state_dict(), self.model_path)
        print(f"Model saved at {self.model_path}")

    def load_model(self):
        _require_torch()
        self.load_state_dict(torch.load(self.model_path))
        self.eval()
        print(f"Model loaded from {self.model_path}")
