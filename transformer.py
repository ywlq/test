import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.simplefilter("ignore")
print(torch.__version__)

class WordEmbedding(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(WordEmbedding,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
    
    def forward(self,x):
        out=self.embed(x)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        super(PositionalEmbedding,self).__init__()
        self.embed_dim=embed_model_dim

        pe=torch.zeros(max_seq_len,self.embed_dim)

        for pos in range(max_seq_len):
            for i in range(0,embed_model_dim,2):
                pe[pos,i]=math.sin(pos/(10000**((2*i)/self.embed_dim)))
                pe[pos,i+1]=math.cos(pos/(10000**((2*(i+1))/self.embed_dim)))
        pe=pe.unsqueeze(0) #(batch_size,seq_len,embed_dim)
        self.register_buffer('pe',pe)

    def forward(self,x):
        #x:(batch_size,seq_len,embed_dim)
        x=x*math.sqrt(self.embed_dim)
        seq_len=x.size(1)
        x=x+torch.autograd.Variable(self.pe[:,:seq_len],requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim=512,n_heads=8):
        super(MultiHeadAttention,self).__init__()
        #embed_dim=d_model
        self.embed_dim=embed_dim
        self.heads=n_heads
        self.single_head_dim=int(self.embed_dim/self.heads)

        self.query_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.value_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.key_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)

        self.out=nn.Linear(self.embed_dim,self.embed_dim)

    def forward(self,key,query,value,mask=None):
        batch_size=key.size(0)
        seq_length=key.size(1)
        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query=query.size(1)
        #(32x10x8x64)
        key=key.view(batch_size,seq_length,self.heads,self.single_head_dim)
        query=query.view(batch_size,seq_length_query,self.heads,self.single_head_dim)
        value=value.view(batch_size,seq_length,self.heads,self.single_head_dim)
        #(32x10x8x64)
        k=self.key_matrix(key)
        q=self.query_matrix(query)
        v=self.value_matrix(value)
        # (32 x 8 x 10 x 64)
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        #(32 x 8 x 64 x 10)
        k_adjusted=k.transpose(-1,-2)
        #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = (32x8x10x10)
        product=torch.matmul(q,k_adjusted)

        if mask is not None:
            product=product.masked_fill(mask==0, float("-1e20"))

        product=product/math.sqrt(self.single_head_dim)
        scores=F.softmax(product,dim=-1)
        #(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)
        scores=torch.matmul(scores,v)
        concat=scores.transpose(1,2).contiguous().view(batch_size,seq_length_query,self.embed_dim)

        output=self.out(concat)

        return output

class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):
        super(TransformerBlock,self).__init__()

        self.attention=MultiHeadAttention(embed_dim,n_heads)
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)

        self.feed_forword=nn.Sequential(
            nn.Linear(embed_dim,expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor*embed_dim,embed_dim)
        )

        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.2)
    
    def forward(self,key,query,value):
        attention_out=self.attention(key,query,value)
        attention_residual_out=attention_out+value
        norm1_out=self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out=self.feed_forword(norm1_out)
        feed_fwd_residual_out=feed_fwd_out+norm1_out
        norm2_out=self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self,seq_len,vocab_size,embed_dim,num_layers=2,expansion_factor=4,n_heads=8):
        super(TransformerEncoder,self).__init__()

        self.embedding_layer=WordEmbedding(vocab_size,embed_dim)
        self.positional_encoder=PositionalEmbedding(seq_len,embed_dim)
        self.layers=nn.ModuleList([TransformerBlock(embed_dim,expansion_factor,n_heads) for i in range(num_layers)])
    
    def forward(self,x):
        embed_out=self.embedding_layer(x)
        out=self.positional_encoder(embed_out)
        for layer in self.layers:
            out=layer(out,out,out) #QKV输入时是一个东西
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):
        super(DecoderBlock,self).__init__()

        self.attention=MultiHeadAttention(embed_dim,n_heads=8)
        self.norm=nn.LayerNorm(embed_dim)
        self.dropout=nn.Dropout(0.2)
        self.transformer_block=TransformerBlock(embed_dim,expansion_factor,n_heads)
    
    def forward(self,key,query,x,mask):
        attention=self.attention(x,x,x,mask=mask)
        value=self.dropout(self.norm(attention+x))
        out=self.transformer_block(key,query,value)  
        return out
    
    # def forward(self,key,x,value,mask):
    #     attention=self.attention(x,x,x,mask=mask)
    #     query=self.dropout(self.norm(attention+x))
    #     out=self.transformer_block(key,query,value)  
    #     return out

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder,self).__init__()
        self.word_embedding=WordEmbedding(target_vocab_size,embed_dim)
        self.position_embedding=PositionalEmbedding(seq_len,embed_dim)

        self.layers=nn.ModuleList([DecoderBlock(embed_dim,expansion_factor,n_heads) for _ in range(num_layers)])

        self.fc_out=nn.Linear(embed_dim,target_vocab_size)
        self.dropout=nn.Dropout(0.2)

    def forward(self,x,enc_out,mask):
        x=self.word_embedding(x)
        x=self.position_embedding(x)
        x=self.dropout(x)

        for layer in self.layers:
            x=layer(enc_out,x,enc_out,mask)
        
        out=F.softmax(self.fc_out(x))
        return out

class Transformer(nn.Module):
    def __init__(self,embed_dim,src_vocab_size,target_vocab_size,seq_length,num_layers=2,expansion_factor=4,n_heads=8):
        super(Transformer,self).__init__()

        self.target_vocab_size=target_vocab_size
        self.encoder=TransformerEncoder(seq_length,src_vocab_size,embed_dim,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)
        self.decoder=TransformerDecoder(target_vocab_size,embed_dim,seq_length,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)

    def mask_trg_mask(self,trg):
        batch_size,trg_len=trg.shape

        trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(batch_size,1,trg_len,trg_len) #tril为下三角函数
        return trg_mask
    
    def decode(self,src,trg):
        trg_mask=self.mask_trg_mask(trg)
        enc_out=self.encoder(src)
        out_lables=[]
        batch_size,seq_len=src.shape[0],src.shape[1]
        out=trg

        for i in  range(seq_len):
            out=self.decoder(out,enc_out,trg_mask)
            out=out[:,-1,:]  # 提取输出序列中最后一个词元的所有特征
            out=out.argmax(-1) #计算最后一个维度上的最大值索引
            out_lables.append(out.item())
            out=torch.unsqueeze(out,axis=0)
        
        return out_lables

    def forward(self,src,trg):
        trg_mask=self.mask_trg_mask(trg)
        enc_out=self.encoder(src)
        outputs=self.decoder(trg,enc_out,trg_mask)

        return outputs
    
if __name__ == '__main__':
    src_vocab_size = 11
    target_vocab_size = 11
    num_layers = 6
    seq_length = 12

    # let 0 be sos(start operation signal) token and 1 be eos(end operation signal) token
    src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],
                        [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
    target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],
                           [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])

    print(src.shape, target.shape)
    model = Transformer(embed_dim=512, src_vocab_size=src_vocab_size,
                        target_vocab_size=target_vocab_size, seq_length=seq_length,
                        num_layers=num_layers, expansion_factor=4, n_heads=8)
    print(model)
    out = model(src, target)
    print(out)
    print(out.shape)
    print("=" * 50 )
    # inference
    src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1]])
    trg = torch.tensor([[0]])
    print(src.shape, trg.shape)
    out = model.decode(src, trg)
    print(out)
