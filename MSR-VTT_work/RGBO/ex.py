import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, word_num, dim ):
        super().__init__()
        self.dim = dim
        
        # word_num行, dim列のゼロベクトルのpe(positional encoding)の作成
        pe = torch.zeros(word_num, dim)
        for pos in range(word_num):
            for i in range(0, dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/dim)))

        # 行列[ [ ...],[....],,, ]を更に囲む（Tensor化)
        pe = pe.unsqueeze(0)
    
    def forward(self, data):
        data = data * math.sqrt(self.dim) # ちょっと値を大きくしてやる
        seq_len = data.size(1) # dataのワード数を取り出す
        data = data + torch.tensor(self.pe[:,:seq_len], requires_grad=False) # 足し込む処理
        return data
    
class MultiHeadSelfAttention( nn.Module ):
    def __init__( self, dimention, num_head, dropout = 0.1):
        super().__init__()

        assert dimention % num_head == 0
        self.dim = dimention
        self.num_head = num_head 
        self.head_dim = self.dim // self.num_head

        self.q_linear = nn.Linear( self.dim, self.dim )
        self.k_linear = nn.Linear(self.dim,self.dim )
        self.v_linear = nn.Linear(self.dim,self.dim )        
        self.out = nn.Linear(self.dim,self.dim )    

        self.dropout = nn.Dropout( dropout )

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]) )


    # q = [batch size, query len, hid dim]
    # k = [batch size, key len, hid dim]
    # v = [batch size, value len, hid dim]
    def foward( self, q, k, v, mask =None ):
        batch_size = q.size( 0 )

        Q = self.q_linear(q) # [batch size, query len, dimention]
        K = self.k_linear(k) 
        V = self.v_linear(v) 

        Q = Q.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        self.k_linear( k ).view( batch_size, -1, self.num_head, self.head_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax( energy, dim = -1 ) # [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V) # [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.dim) # [batch size, query len, dim]

        return self.out( x ), attention # self.out(x) == [batch size, query len, hid dim]
    
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

class EncoderBlock(nn.Module):
    def __init__(self, dimention, n_heads, dropout ):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention( dimention, n_heads, dropout )
        self.self_attn_layer_norm = nn.LayerNorm(dimention)
        self.ff_layer_norm = nn.LayerNorm(dimention)
        self.feadforward = FeedForward(dimention)
        
        self.dropout_1 = nn.Dropout(dropout)                
        self.dropout_2 = nn.Dropout(dropout)                

    def forward( self, x, src_mask ):
        #x = [batch size, src len, dim]
        #src_mask = [batch size, 1, 1, src len] 
        new_x, _ = self.self_attention( x, x, x, src_mask)  
        new_x = self.self_attn_layer_norm(x + self.dropout_1(new_x)) #src_x = [batch size, src len, dim]
        out_x = self.feadforward(new_x)        
        out_x = self.ff_layer_norm(x + self.dropout_2(out_x)) #out_x = [batch size, src len, dim]
        return out_x

class Encoder(nn.Module):
    def __init__(self, vocab_size, dimention, Nx, n_heads, dropout = 0.1, max_word_len = 100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dimention)
        self.pe = PositionalEncoder( max_word_len, dimention)        
        self.blocks = nn.ModuleList([ EncoderBlock( dimention,  n_heads, dropout ) for _ in range(Nx) ] )
        self.norm = Norm(dimention)

    #src = [batch size, src len]
    #src_mask = [batch size, 1, 1, src len]
    def forward(self, src, src_mask):
        x = self.embed(src)
        x = self.pe(x)        
        for encoder_block in self.blocks:
            x =encoder_block(x, src_mask) #src = [batch size, src len, hid dim]        
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, dimention,  n_heads, dropout ):                 
        super().__init__()        
        self.norm1 = nn.LayerNorm(dimention)
        self.norm2 = nn.LayerNorm(dimention)
        self.norm3 = nn.LayerNorm(dimention)
        self.dropout = nn.Dropout(dropout)

        self.attention_self = MultiHeadSelfAttention(dimention, n_heads, dropout)
        self.attention_encoder = MultiHeadSelfAttention(dimention, n_heads, dropout)
        self.feedforward =FeedForward(dimention)                                                                     

    #x = [batch size, target len, hid dim]
    #enc_src = [batch size, src len, hid dim]
    #target_mask = [batch size, 1, target len, target len]
    #src_mask = [batch size, 1, 1, src len]
    def forward(self, x, x_from_encoder, target_mask, src_mask):        
        _target, _ = self.attention_self(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(_target))   #x = [batch size, target len, hid dim]                    
        _target, attention = self.attention_encoder(x, x_from_encoder, x_from_encoder, src_mask)         
        x = self.norm2(x + self.dropout(_target)) # target = [batch size, target len, hid dim]        
        _target = self.feedforward(x)        
        x = self.norm3(x + self.dropout(_target)) # target = [batch size, target len, hid dim], 
        return x, attention # attention = [ batch size, n heads, target len, src len ]         

class Decoder(nn.Module):
    def __init__(self, vocab_size, dimention, n_layers, n_heads, dropout = 0.1, max_word_len = 100):
        super().__init__()
        self.embed_vocab = nn.Embedding( vocab_size, dimention ) # 
        self.pe = PositionalEncoder( max_word_len, dimention)        
        self.blocks = nn.ModuleList( [ DecoderBlock(dimention, n_heads  , dropout ) for _ in range(n_layers) ] )
        self.scale = torch.sqrt(torch.FloatTensor([dimention]))
        self.norm = Norm(dimention)

    #trg = [batch size, trg len]
    #enc_src = [batch size, src len, dimention]
    #trg_mask = [batch size, 1, trg len, target len]
    #src_mask = [batch size, 1, 1, src len]                
    def forward(self, target, x_from_encoder, trg_mask, src_mask):        
        x = self.embed_vocab( target )
        x = self.pe( x )
        for decoder_block in self.blocks:
            x, attention = decoder_block(x, x_from_encoder, trg_mask, src_mask) #target  = [batch size, trg len, hid dim], attention = [batch size, n heads, trg len, src len]
        output = self.norm(x) #output = [batch size, trg len, output dim]
        return output , attention    

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads )
        self.decoder = Decoder(trg_vocab, d_model, N, heads )
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, target, src_mask, trg_mask):
        output_encoder = self.encoder(src, src_mask)
        output_decoder = self.decoder(target, output_encoder, trg_mask, src_mask )
        output = self.out(output_decoder)
        return output     