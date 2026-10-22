"""
training/ocr_correction_model.py
Character-level Seq2Seq LSTM for Tamil OCR text correction.
  Encoder: noisy OCR text → context vector
  Decoder: context vector → clean Tamil text
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # Project bidirectional hidden to hidden_dim for decoder
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, src):
        # src: (B, T)
        embedded = self.embedding(src)                    # (B, T, E)
        outputs, (hidden, cell) = self.lstm(embedded)     # outputs: (B, T, H*2)

        # Merge bidirectional: take last layer forward+backward
        # hidden: (num_layers*2, B, H)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        cell   = cell.view(self.num_layers, 2, -1, self.hidden_dim)

        # Concat forward & backward for each layer
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # (L, B, H*2)
        cell   = torch.cat([cell[:, 0],   cell[:, 1]],   dim=2)

        # Project to hidden_dim
        hidden = torch.tanh(self.fc_h(hidden))  # (L, B, H)
        cell   = torch.tanh(self.fc_c(cell))

        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, enc_dim: int):
        super().__init__()
        self.attn    = nn.Linear(hidden_dim + enc_dim, hidden_dim)
        self.v       = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: (B, H), enc_outputs: (B, T, H*2)
        B, T, _ = enc_outputs.shape
        h = dec_hidden.unsqueeze(1).repeat(1, T, 1)       # (B, T, H)
        energy = torch.tanh(self.attn(torch.cat([h, enc_outputs], dim=2)))
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 enc_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim, enc_dim)
        self.lstm = nn.LSTM(
            embed_dim + enc_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out  = nn.Linear(hidden_dim + enc_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, hidden, cell, enc_outputs):
        # tgt_token: (B,)
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))  # (B,1,E)
        context, _ = self.attention(hidden[-1], enc_outputs)             # (B, enc_dim)
        lstm_in = torch.cat([embedded, context.unsqueeze(1)], dim=2)    # (B,1,E+enc_dim)
        output, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        output  = output.squeeze(1)                                       # (B, H)
        pred    = self.fc_out(torch.cat([output, context, embedded.squeeze(1)], dim=1))
        return pred, hidden, cell


class Seq2SeqCorrector(nn.Module):
    """
    Full Seq2Seq model with attention for OCR correction.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        enc_out_dim = hidden_dim * 2  # bidirectional
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim,
                               enc_out_dim, num_layers)
        self.vocab_size = vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio: float = 0.5):
        """
        src: (B, T_src)
        tgt: (B, T_tgt)  — includes SOS token at position 0
        Returns: (B, T_tgt, vocab_size)
        """
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.vocab_size, device=src.device)

        enc_outputs, hidden, cell = self.encoder(src)
        dec_input = tgt[:, 0]  # SOS

        for t in range(1, T):
            pred, hidden, cell = self.decoder(dec_input, hidden, cell, enc_outputs)
            outputs[:, t] = pred
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else pred.argmax(dim=1)

        return outputs

    def greedy_decode(self, src: torch.Tensor, sos_idx: int, eos_idx: int,
                      max_len: int = 128) -> list:
        """Inference: greedy decoding for a single sample."""
        self.eval()
        with torch.no_grad():
            enc_outputs, hidden, cell = self.encoder(src.unsqueeze(0))
            dec_input = torch.tensor([sos_idx], device=src.device)
            result = []
            for _ in range(max_len):
                pred, hidden, cell = self.decoder(dec_input, hidden, cell, enc_outputs)
                top = pred.argmax(dim=1)
                if top.item() == eos_idx:
                    break
                result.append(top.item())
                dec_input = top
        return result


def get_seq2seq(vocab_size: int, device: str = "cpu") -> Seq2SeqCorrector:
    model = Seq2SeqCorrector(vocab_size)
    model.to(device)
    return model
