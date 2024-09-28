from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class Embedder(nn.Module):

    def __init__(self, voc_size, d_model):
        super(Embedder, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedder = nn.Embedding(voc_size, d_model)

    def forward(self, x):
        x = self.embedder(x)
        x = x + np.sqrt(self.d_model)
        return x

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(1, d_model, 2)
        evens = np.arange(0, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, evens] = np.sin(pos / (10000 ** (evens / d_model)))
            pos_enc_mat[pos, odds] = np.cos(pos / (10000 ** (odds / d_model)))
        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)

    def forward(self, x):
        B, S, d_model = x.shape
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)

        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, H, dout_p):
        super(MultiHeadAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.H = H
        self.d_k = d_model // H
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dout_p)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, q, k, v, mask=None):
        batch_size = Q.size(0)

        Q = self.q_linear(q) # [batch size, query len, dim]
        K = self.k_linear(k)
        V = self.v_linear(v)

        Q = Q.view(batch_size, -1, self.H, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.H, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.H, self.d_k).permute(0, 2, 1, 3)

        self.k_linear(k).view(batch_size, -1, self.H, self.d_k)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.out(x), attention

class Norm(nn.Module):
    
    def __init__(self, d_model, eps):
        super(Norm, self).__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x-x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True)+self.eps)+self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, dimention, d_ff=2048, dropout = 0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(dimention, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, dimention)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model, H, dout_p,d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, H, dout_p)
        self.self_atten_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dout_p)
        self.dropout_2 = nn.Dropout(dout_p)

    def forward(self, x, src_mask):
        # x = [batch size, src len, dim]
        new_x, _ = self.self_attention(x, x, x, src_mask)
        # src_mask = [batch size, 1, 1, src len]
        new_x = self.self_atten_layer_norm(x + self.dropout_1(new_x))
        out_x = self.feed_forward(new_x)
        out_x = self.ff_layer_norm(x + self.dropout_2(out_x))
        # out_x = [batch size, src len, dim]
        return out_x

class Encoder(nn.Module):

    def __init__(self, d_model, N, H, dout_p, max_len):
        super(Encoder, self).__init__()
        self.pe = PositionalEncoder(d_model, dout_p, max_len)
        self.encoder = nn.ModuleList([ EncoderLayer(d_model, H, dout_p) for _ in range(N)])
        self.norm = Norm(d_model)

    def forward(self, src, src_mask):
        x = self.pe(src)
        for encoder_l in self.encoder:
            # src = [batch size, src len, hid dim]
            x = encoder_l(x, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, d_model, H, dout_p):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dout_p)

        self.attention_self = MultiHeadAttention(d_model, H, dout_p)
        self.attention_encoder = MultiHeadAttention(d_model, H, dout_p)
        self.feed_forward = FeedForward(d_model)

    def forward(self, x, x_from_encoder, target_mask, src_mask):
        _target, _ = self.attention_self(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(_target))
        _target, attention = self.attention_encoder(x, x_from_encoder, x_from_encoder, src_mask)
        x = self.norm2(x + self.dropout(_target))
        _target = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_target))

        return x, attention
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, max_len, d_model, N, H, dout_p, ):
        super(Decoder, self).__init__()
        self.embed_vocab = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dout_p, max_len)
        self.decoder = nn.ModuleList(DecoderLayer(d_model, H, dout_p) for _ in range(N))
        self.scale = torch.sqrt(torch.FloatTensor(([d_model])))
        self.norm =Norm(d_model)

    def forward(self, target, x_from_encoder, trg_mask, src_mask):
        x = self.embed_vocab(target)
        x = self.pe(x)
        for decoder_l in self.decoder:
            x, attention = decoder_l(x, x_from_encoder, trg_mask, src_mask)
        output = self.norm(x)

        return output, attention

class Generator(nn.Module):
    
    def __init__(self, ):
        pass

    def forward(self):
        pass

class VideoCaptioningTransformer(nn.Module):
    
    def __init__(self, src_video, trg_vocab, dout_p, d_model, voc_size, H, d_ff, N):
        super(VideoCaptioningTransformer, self).__init__()
        self.encoder = Encoder(d_model,  )

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self):
        pass
