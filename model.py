from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CalculateQKV(nn.Module):
    def __init__(self,dim, heads,dim_head):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        return q,k,v
    
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,q,k,v,mask=None):
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out




class MultiheadEncoder(nn.Module):
    def __init__(self, dim, heads, dim_head, fuse_dim, dropout=0.):
        super().__init__()
        self.CalculateQKV = CalculateQKV(dim, heads,dim_head)
        self.Attention = Attention(dim, heads, dim_head, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = Residual(PreNorm(dim, FeedForward(dim, fuse_dim, dropout=dropout)))

    def forward(self, x, mask=None):
        #print("decoder input: ",x.shape)
        q,k,v = self.CalculateQKV(x)
        att_score = self.Attention(q,k,v,mask=mask)
        x_att = self.norm(att_score)
        x = x_att + x
        output = self.ffn(x)
        return output
    

  
class Encoder_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, fuse_dim, dropout=0.):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.pos_embedding_eeg = nn.Parameter(torch.randn(1, 6, dim))
        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, dim))
        self.layer_stack = nn.ModuleList([
            MultiheadEncoder(dim, heads, dim_head, fuse_dim)
            for _ in range(depth)])
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, fuse_dim)
        )

    def forward(self, x, mask=None):
        b_g, t_g, d_g = x.shape
        x = x.reshape(b_g*t_g, d_g) # 160*310 (32*5)
        x = self.bn(x)
        x = x.reshape(b_g, t_g, d_g )

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token_eeg, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding_eeg[:, :(n + 1)]
        #print("dim after pos: ",x.shape)
        for enc_layer in self.layer_stack:
            x = enc_layer(x, mask=mask)
        # for multiAttn in self.layers:
        #     x = multiAttn(x, mask=mask)               #32*5*310
        out = self.mlp(x)           #32*5*15   fuse_dim 15
        b,w,d = out.shape
        #print("output window: ",w)
        #print("output dim: ",d)
        final = out.reshape(b, w*d) #32*75
        return final


class Decoder_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, hidden_dim,dropout=0.):
        super().__init__()
        #self.tranf = nn.Linear(hidden_dim*2, dim)
        self.layer_stack = nn.ModuleList([
                    MultiheadEncoder(hidden_dim, heads, dim_head, dim)
                    for _ in range(depth)])
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        for multiAttn in self.layer_stack:
            enc_output = multiAttn(enc_output)  
        dec_output = self.mlp(enc_output)
        return dec_output[:,1:,:]

class Unify_Dim(nn.Module):
    def __init__(self, dim, target_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.linear = nn.Linear(dim, target_dim)
    
    def forward(self, x):
        b_g, t_g, d_g = x.shape
        x = x.reshape(b_g*t_g, d_g) # 160*310 (32*5)
        x = self.bn(x)
        x = x.reshape(b_g, t_g,d_g)
        x = self.linear(x)
        return x


class CrossMultiheadEncoder(nn.Module):
    def __init__(self, dim, heads, dim_head, fuse_dim, dropout=0.):
        super().__init__()

        self.CalculateQKV_eeg = CalculateQKV(dim, heads,dim_head)
        self.CalculateQKV_eye = CalculateQKV(dim, heads,dim_head)
        self.Attention = Attention(dim, heads, dim_head, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = Residual(PreNorm(dim, FeedForward(dim, fuse_dim, dropout=dropout)))

    def forward(self, eeg,eye, mask=None):
        q_eeg,k_eeg,v_eeg = self.CalculateQKV_eeg(eeg)
        q_eye,k_eye,v_eye = self.CalculateQKV_eye(eye)
        att_score_eye = self.Attention(q_eeg,k_eye,v_eye,mask=mask)
        att_score_eeg = self.Attention(q_eye,k_eeg,v_eeg,mask=mask)
        x_att = self.norm(att_score_eye+att_score_eeg)
        x = x_att + eye + eeg  # TODO which input signal should residual network use?
        output = self.ffn(x)
        return output


class Cross_Attention_Layer_2(nn.Module):
    def __init__(self, dim_eeg, dim_eye, depth, heads, dim_head, fuse_dim, dropout=0.):
        super().__init__()
        target_dim = fuse_dim*2
        self.Unify_Dim_eeg = Unify_Dim(dim_eeg, target_dim)
        self.Unify_Dim_eye = Unify_Dim(dim_eye, target_dim)
        self.pos_embedding_eeg = nn.Parameter(torch.randn(1, 6, dim_eeg))
        self.cls_token_eeg = nn.Parameter(torch.randn(1, 1, dim_eeg))
        self.pos_embedding_eye = nn.Parameter(torch.randn(1, 6, dim_eye))
        self.cls_token_eye = nn.Parameter(torch.randn(1, 1, dim_eye))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(CrossMultiheadEncoder(target_dim,heads, dim_head, fuse_dim))
        self.mlp = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, fuse_dim)
        )


    def forward(self, eeg, eye, mask=None):
        #print("input dim: ",eeg.shape) #32*5*310
        b, n, _ = eeg.shape
        cls_tokens = repeat(self.cls_token_eeg, '() n d -> b n d', b=b)
        eeg = torch.cat((cls_tokens, eeg), dim=1)
        eeg += self.pos_embedding_eeg[:, :(n + 1)]

        b, n, _ = eye.shape
        cls_tokens = repeat(self.cls_token_eye, '() n d -> b n d', b=b)
        eye = torch.cat((cls_tokens, eye), dim=1)
        eye += self.pos_embedding_eye[:, :(n + 1)]

        eeg = self.Unify_Dim_eeg(eeg)
        eye = self.Unify_Dim_eye(eye)
        #print("unified dim: ", eeg.shape) #32*5*26

        for crossAtt in self.layers:
            x = crossAtt(eeg,eye)
        #print("after cross attention dim: ",x.shape) # 32*5*30
        out = self.mlp(x)           #32*5*15   fuse_dim 15
        #print("mlp dim: ",out.shape)
        b,w,d = out.shape
        final = out.reshape(b, w*d) #32*75
        #print("final dim: ",final.shape)
        return final    
    

    
class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        #    nn.Tanh(),
         #   nn.ReLU()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class Classifier_Emotion(nn.Module):
    def __init__(self, in_size, output_dim ,dropout=0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(in_size, in_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_size*2, output_dim),
        )

    def forward(self, x):
        normed = self.norm(x)
        dropped = self.drop(normed)
        out = self.ffn(dropped)
        return out


