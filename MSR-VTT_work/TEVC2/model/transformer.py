from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
        mask = torch.ones(1, size, size)
        mask = torch.tril(mask, 0)
        
        return mask.byte() # ([1, size, size])

def mask(src, trg, pad_idx):
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)
    
    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)
        return src_mask, trg_mask
    else:
        return src_mask

class Embedder(nn.Module):
    
    def __init__(self, voc_size, d_model):
        super(Embedder, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedder = nn.Embedding(voc_size, d_model)

    # input: [Batch size, seq len, feat_dim(1024)]
    def forward(self, x):
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        
        return x # (B, seq_len, d_model)

class PositionalEncoder(nn.Module):

    def __init__(self, seq_len, d_model, dout_p):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        
        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))
        
        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)

    # input: [Batch size, seq len, feat_dim(1024)]
    def forward(self, x):
        B, seq_len, d_model = x.shape
        x = x + self.pos_enc_mat[:, :seq_len, :].type_as(x)
        x = self.dropout(x)
        return x    # Shape: [Batch size, seq len, feat_dim]

class ResidualConnection(nn.Module):
    
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)
        
    def forward(self, x, sublayer): # [(B, seq_len, d_model), attention or feed forward]
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        
        return x + res

class Norm(nn.Module):
    def __init__(self, dimention, eps = 1e-6):
        super().__init__()
            
        self.size = dimention        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # todo dropout?
        
    def forward(self, x): # x - (B, seq_len, d_model)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x # x - (B, seq_len, d_model)

class MultiheadAttention(nn.Module):
    
    def __init__(self, d_model, H, dout_p):
        super(MultiheadAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.num_head = H
        self.head_dim = self.d_model // self.num_head

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dout_p)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, feat_dim = q.shape

        Q = self.q_linear(q) # [batch size, query len, dimention]
        K = self.k_linear(k) 
        V = self.v_linear(v) 

        Q = Q.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        self.k_linear( k ).view( batch_size, -1, self.num_head, self.head_dim)
        if mask is not None:
            # the same mask for all heads
            mask = mask.unsqueeze(1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device) # [batch size, n heads, query len, key len]

        
        if mask is not None:
            energy = energy.masked_fill_(mask == 0, -1e10)

        attention = torch.softmax( energy, dim = -1 ) # [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V) # [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model) # [batch size, query len, dim]
        x = self.out(x)

        return x

class PureEncoderLayer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff):
        super(PureEncoderLayer, self).__init__()
        self.res1 = ResidualConnection(d_model, dout_p)
        self.res2 = ResidualConnection(d_model, dout_p)
        self.self_attention = MultiheadAttention(d_model, H, dout_p)
        self.fcn = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, x, src_mask):
        # Self-MultiHeadAttention
        # ResidualConnection(= Add + Layer Normalization)
        # Position-wise Fully-Connected Network
        sublayer0 = lambda x: self.self_attention(x, x, x, src_mask)
        sublayer1 = self.fcn
        x = self.res1(x, sublayer0)
        x = self.res2(x, sublayer1)

        return x

class CrossEncoderLayer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff):
        super(CrossEncoderLayer, self).__init__()
        self.res1 = ResidualConnection(d_model, dout_p)
        self.res2 = ResidualConnection(d_model, dout_p)
        self.res3 = ResidualConnection(d_model, dout_p)
        self.self_attention = MultiheadAttention(d_model, H, dout_p)
        self.cross_attention = MultiheadAttention(d_model, H, dout_p)
        self.fcn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x_flow, x_rgb, src_mask):
        sublayer0 = lambda x_rgb: self.self_attention(x_rgb, x_rgb, x_rgb, src_mask)
        sublayer1 = lambda x_rgb: self.cross_attention(x_rgb, x_flow, x_flow)
        # sublayer1 = lambda x_rgb: self.cross_attention(x_flow, x_rgb, x_rgb)
        sublayer2 = self.fcn
        x = self.res1(x_rgb, sublayer0)
        x = self.res2(x, sublayer1)
        x = self.res3(x, sublayer2)

        return x


class Encoder(nn.Module):

    def __init__(self, seq_len, d_model, dout_p, H, N, d_ff, flag):
        super(Encoder, self).__init__()
        self.flag = flag
        self.pe = PositionalEncoder(seq_len, d_model, dout_p)
        if flag == 'pure':
            self.enc = nn.ModuleList([PureEncoderLayer(d_model, dout_p, H, d_ff) for _ in range(N)])
        elif flag == 'cross':
            self.enc = nn.ModuleList([CrossEncoderLayer(d_model, dout_p, H, d_ff) for _ in range(N)])
        self.norm = Norm(d_model)
    
    def forward(self, src1, src2, src1_mask):
        x = self.pe(src1)
        for layer in self.enc:
            if self.flag == 'pure':
                x = layer(x, src1_mask)
            elif self.flag == 'cross':
                x = layer(x, src2, src1_mask)
        
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff):
        super(DecoderLayer, self).__init__()
        self.res1 = ResidualConnection(d_model, dout_p)
        self.res2 = ResidualConnection(d_model, dout_p)
        self.res3 = ResidualConnection(d_model, dout_p)
        self.self_attention = MultiheadAttention(d_model, H, dout_p)
        self.enc_attention = MultiheadAttention(d_model, H, dout_p)
        self.fcn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, memory, src_mask, trg_mask):
        sublayer0 = lambda x: self.self_attention(x, x, x, trg_mask)
        sublayer1 = lambda x: self.enc_attention(x, memory, memory, src_mask)
        sublayer2 = self.fcn
        x = self.res1(x, sublayer0)
        x = self.res2(x, sublayer1)
        x = self.res3(x, sublayer2)

        return x

class Decoder(nn.Module):

    def __init__(self, seq_len, voc_size, d_model, dout_p, H, N, d_ff):
        super(Decoder, self).__init__()
        self.embed = Embedder(voc_size, d_model)
        self.pe = PositionalEncoder(seq_len, d_model, dout_p)
        self.dec = nn.ModuleList([DecoderLayer(d_model, dout_p, H, d_ff) for _ in range(N)])
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))
        self.norm = Norm(d_model)
    
    def forward(self, trg, memory, trg_mask, src_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for layer in self.dec:
            x = layer(x, memory, src_mask, trg_mask)
        output = self.norm(x)
        return output

class Generator(nn.Module):

    def __init__(self, d_model, voc_size, dout_p):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))

        return F.log_softmax(x, dim=-1)
        
class RGBOnlyTransformer(nn.Module):

    def __init__(self, trg_vocab_size, d_model, H, N, d_ff, dout_p, seq_len):
        super(RGBOnlyTransformer, self).__init__()
        self.pure_encoder = Encoder(seq_len, d_model, dout_p, H, N, d_ff, 'pure')
        self.decoder = Decoder(seq_len, trg_vocab_size, d_model, dout_p, H, N, d_ff)
        self.generator = Generator(d_model, trg_vocab_size, dout_p)

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print('This model is RGBOnlyTransformer model!')

    def forward(self, src_rgb, trg, src_rgb_mask, trg_mask):
        memory = self.pure_encoder(src_rgb, src_rgb, src_rgb_mask)
        out_decoder = self.decoder(trg, memory, trg_mask, src_rgb_mask)
        out = self.generator(out_decoder)

        return out

class TEVCTransformer(nn.Module):

    def __init__(self, trg_vocab_size, d_model, H, N, d_ff, dout_p, seq_len):
        super(TEVCTransformer, self).__init__()
        self.pure_encoder = Encoder(seq_len, d_model, dout_p, H, N, d_ff, 'pure')
        self.cross_encoder = Encoder(seq_len, d_model, dout_p, H, N, d_ff, 'cross')
        self.decoder = Decoder(seq_len, trg_vocab_size, d_model, dout_p, H, N, d_ff)
        # self.out = nn.Linear()
        self.generator = Generator(d_model, trg_vocab_size, dout_p)

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_rgb, src_flow, trg, masks):
        src_rgb_mask, src_flow_mask, trg_mask = masks
        memory_flow = self.pure_encoder(src_flow, src_flow, src_flow_mask)
        memory_cross_encoder = self.cross_encoder(src_rgb, memory_flow, src_rgb_mask)
        out_decoder = self.decoder(trg, memory_cross_encoder, trg_mask, src_rgb_mask)
        out = self.generator(out_decoder)

        return out
    

