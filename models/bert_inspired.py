import math
import torch
import torch.nn as nn
from layers.transformer_encdec import Encoder, EncoderLayer
from layers.attention import FullAttention, AttentionLayer
from layers.embed import DataEmbedding


class BertInspired(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self, config):
        super(BertInspired, self).__init__()
        self.output_attention = config.output_attention

        # Embedding
        c_in = config.enc_in
        if type(config.enc_in) is tuple:
            c_in = math.prod(config.enc_in)
        self.enc_embedding = DataEmbedding(c_in, config.d_model, config.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, config.factor, attention_dropout=config.dropout,
                                      output_attention=config.output_attention), config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.fc = nn.Linear(config.d_model, config.c_out)
        self.relu = nn.ReLU()
        

    def forward(self, x_enc, enc_self_mask=None):
        x_enc  = x_enc.reshape(*x_enc.shape[:-2], -1)

        enc_emb = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_emb, attn_mask=enc_self_mask)

        # print ("enc out shape: ", enc_out.shape)
        out = self.fc(enc_out) #out[:, -1, :]
        # print ("linear out shape: ", out.shape)
        out = self.relu(out)

        if self.output_attention:
            return out, attns
        else:
            return out
