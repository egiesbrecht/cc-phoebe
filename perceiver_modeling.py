import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, NamedTuple
import copy
import numpy as np
import random
from einops import rearrange, einsum

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from xpos_relative_position import XPOS
from rotary_embeddings import RotaryEmbedding
from contextual_position_embeddings import CoPE
from model_training import _shift_right, emb2idx, num_parameters, num_trainable_parameters
from rmsnorm import RMSNorm

from transformers import PreTrainedModel, PretrainedConfig
from transformers.pytorch_utils import apply_chunking_to_forward


def get_act_func(name, dim=None):
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "mish":
        return nn.Mish()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "softmax":
        return nn.Softmax(dim=dim if dim is not None else -1)
    if name == "softplus":
        return nn.Softplus()
    if name == "none" or name == None:
        return None
    else:
        raise ValueError(f"unknown activation function '{name}'")


def _3d_group_mask(group_mask):
    B, T = group_mask.shape
    out = []
    for i in range(0, T):
        n = torch.cat((
            torch.zeros(B, i).to(group_mask.device),
            group_mask[:, :-(i)] if i > 0 else group_mask
        ), 1)
        out.append(n)
    return torch.stack(out, -1)


def _get_D(gamma, dim1, dim2=None):
    if dim2 is None:
        dim2 = dim1
    n = torch.arange(dim1).unsqueeze(1)
    m = torch.arange(dim2).unsqueeze(0)
    # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
    D = (gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
    # fill the NaN with 0
    D[D != D] = 0
    return D


class PerceiverOutput:
    def __init__(self, logits=None, S=None, C=None, loss=None, aux_loss=None):
        self.logits = logits
        self.encoder_hidden_state = None
        self.S = S
        self.C = C
        self.loss = loss
        self.aux_loss = aux_loss


class PerceiverConfig(PretrainedConfig):
    model_type = "pci4"

    def __init__(
        self,
        hidden_size: int = 1024,
        input_size: Optional[int] = None,
        
        num_layers: int = 1,
        num_heads: int = 1,
        num_kv_heads: Optional[int] = None,
        
        rms_norm_eps: float = 1e-05,
        
        rope_dim: int = 16,

        conv_kernel_size: int = 4,
        
        hidden_dropout_prob: float = 0.1,
        vocab_size: int = 30522,
        num_labels: int = 2,
        max_position_embeddings: int = 512,

        apply_causal_mask: bool = True,
       
        pad_token_id: int = 0,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size if input_size is not None else hidden_size

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        
        self.rms_norm_eps = rms_norm_eps
        
        self.rope_dim = rope_dim
        
        self.conv_kernel_size = conv_kernel_size
        
        self.hidden_dropout_prob = hidden_dropout_prob
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.max_position_embeddings = max_position_embeddings

        self.apply_causal_mask = apply_causal_mask
        
        self.initializer_range = initializer_range


class PerceiverPreTrainedModel(PreTrainedModel):
    config_class = PerceiverConfig
    base_model_prefix = "ci4"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class PerceiverEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(self, input_ids, inputs_embeds=None):
        X = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        #X = self.norm(X)
        #X = self.dropout(X)
        return X


class GLU(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.size = intermediate_size
        self.fi = nn.Linear(input_size, intermediate_size * 2)
        self.fc = nn.Linear(intermediate_size, output_size)
        self.act = nn.SiLU()
        #self.act = nn.Softmax(-1)

    def forward(self, x):
        z, r = self.fi(x).split([self.size, self.size], -1)
        #z = F.softmax(z, 1)
        #z = z.exp()
        o = z * self.act(r)
        return self.fc(o)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PartialSoftmax(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dim: Optional[int]=-1, temperature: Optional[float]=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.temp = temperature
        self.weight = nn.Parameter(torch.randn(self.kernel_size, self.kernel_size) / self.kernel_size)
        self.bias = nn.Parameter(torch.randn(self.kernel_size) / self.kernel_size)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor]=None):
        x_norm = (x / self.temp) #* self.weight.unsqueeze(0)
        #x_exp = torch.exp(x_norm)
        #state = state + x_exp if state is not None else x_exp
        state = x_norm + state if state is not None else x_norm
        x_exp = state.exp()
        x_exp_sum = torch.sum(x_exp, self.dim, keepdim=True)
        out = (x_exp / x_exp_sum) * 2 #+ self.bias
        return out, state


class PartialGumbelSoftmax(PartialSoftmax):
    def __init__(self, dim: Optional[int]=-1, temperature: Optional[float]=1.0):
        super().__init__(dim, temperature)
        self.latent_dim = 30
        self.categorical_dim = 10

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, h_t):
        y = logits + sample_gumbel(logits.size())
        return super().forward(y, h_t)
    
    def gumbel_softmax(self, logits, hard=False, h_t=None):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = gumbel_softmax_sample(logits, h_t)

        if not hard:
            return y.view(-1, self.latent_dim * self.categorical_dim)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, self.latent_dim * self.categorical_dim)


 #########################
 #                       #
 # TODO: implement heads #
 #                       #
 #########################

class PCI5Attention(nn.Module):
    def __init__(self, config, input_size: Optional[int]=None, mask=False):
        super().__init__()
        self.dim = config.hidden_size
        self.mask = mask
        self.input_size = input_size if input_size is not None else self.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        assert self.dim % self.num_heads == 0
        self.head_dim = self.dim // self.num_heads
        assert self.input_size % self.num_kv_heads == 0
        self.kv_dim = self.input_size // self.num_kv_heads
        self.softmax = PartialSoftmax(self.dim, 64, dim=-1, temperature=1)

        self.Q_proj = nn.Linear(self.dim, self.head_dim * self.num_heads)
        self.K_proj = nn.Linear(self.input_size, self.kv_dim * self.num_kv_heads)
        self.V_proj = nn.Linear(self.input_size, self.kv_dim * self.num_kv_heads)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.fc = nn.Linear(self.dim, self.dim)
        self.apply_mask = config.apply_causal_mask

    def _qkv(self, latent_space, input_space):
        Q = self.Q_proj(latent_space)
        K = self.K_proj(latent_space if input_space is None else input_space)
        V = self.V_proj(latent_space if input_space is None else input_space)
        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(K)
        return Q, K, V

    def forward(self, latent_space, input_space=None, h_t=None, mask=True, A_cache=None, A_cache_Q_idx=None):
        B, T_q, C = latent_space.shape
        Q, K, V = self._qkv(latent_space, input_space)

        qk_dot = Q @ K.transpose(-2, -1)
        M = torch.ones(qk_dot.shape[-2:]).triu(1).bool().to(qk_dot.device)
        A = qk_dot
        if self.apply_mask:
            A.masked_fill_(M, float("-inf"))
        #A = F.softmax(A / (self.head_dim ** .5), -1)
        #A, h_t = partial_softmax(A / (self.head_dim ** .5), h_t, -1)
        A, h_t = self.softmax(A / (self.head_dim ** .5), h_t)
        out = self.fc(A @ V)
        return out, h_t, A_cache, A_cache_Q_idx


class ITAttention(nn.Module):
    def __init__(self, config, input_size=None, mask=False):
        super().__init__()
        self.d = config.hidden_size
        self.J = self.d
        self.l_proj = nn.Linear(self.d, self.d)
        self.u_proj = nn.Linear(self.d, self.d)
        self.A_proj = nn.Linear(self.d, self.J)
        self.V_proj = nn.Linear(self.J, self.d)
        self.o_proj = nn.Linear(self.d, self.d)

    def forward(self, latent_space, input_space=None, mask=True, A_cache=None, A_cache_Q_idx=None):
        B, T_q, C = latent_space.shape
        emb_0 = self.l_proj(latent_space)
        l = F.softmax(self.u_proj(emb_0), -1).transpose(-2, -1)
        D = self.A_proj(input_space)
        A = F.softmax(self.V_proj(D), -2)
        w = A @ l
        o = w @ self.o_proj(latent_space)
        return o, A_cache, A_cache_Q_idx


__ATTENTION_CLASS__ = PCI5Attention
#__ATTENTION_CLASS__ = ITAttention

class LatentTransformation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.self_rms = RMSNorm(self.dim, eps=config.rms_norm_eps)
        self.self_proj = nn.Linear(self.dim, self.dim)
        self.self_attn = __ATTENTION_CLASS__(config, mask=True)
        self.glu_rms = RMSNorm(self.dim, eps=config.rms_norm_eps)
        self.glu = GLU(self.dim, config.intermediate_size, self.dim)
    
    def forward(self, latent_space, h_t, A_cache, Q_idx):
        x_self = self.self_rms(latent_space)
        x_self = self.self_proj(x_self)
        latent_space, h_t, A_cache, Q_idx = self.self_attn(x_self, h_t=h_t, mask=True, A_cache=A_cache, A_cache_Q_idx=Q_idx) 
        latent_space += x_self #+ x_latent # latent_space

        #x_glu = self.glu_rms(latent_space)
        #latent_space = self.glu(x_glu) + latent_space

        return latent_space, h_t, A_cache, Q_idx

class CausalConv1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.kernel_size = config.conv_kernel_size
        self.weight = nn.Parameter(torch.randn(self.dim, self.kernel_size) / self.dim)
        self.bias = nn.Parameter(torch.randn(self.dim) / self.dim)
        self.step = 1

    def init_conv_state(self, B):
        return torch.zeros(B, self.dim, self.kernel_size)

    def forward(self, x, conv_state):
        B, T, C = x.shape
        x = x.transpose(-2, -1)
        ret = []
        for t in range(0, T, self.step):
            conv_state = torch.cat((
                conv_state[..., 1:],
                x[..., t : t+self.step]
            ), -1)
            w = self.weight
            y = torch.sum(conv_state * w, -1) + self.bias
            ret.append(y)
        return torch.stack(ret, -1).transpose(-2, -1), conv_state


class cGRU(nn.Module):
    def __init__(self, input_size, hidden_size, apply_input_transformation=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.apply_input_transformation = apply_input_transformation
        self.fz = nn.Linear(input_size, hidden_size)
        self.uz = nn.Linear(hidden_size, hidden_size)
        self.fr = nn.Linear(input_size, hidden_size)
        self.ur = nn.Linear(hidden_size, hidden_size)
        self.fh = nn.Linear(input_size, hidden_size)
        self.uh = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.SiLU()

    def forward(self, x, h=None, chunk_size=1):
        B, T, C = x.shape
        if h is None:
            h = torch.zeros(B, chunk_size, self.hidden_size).to(x.device)
        ret = []
        for t in range(0, T, chunk_size):
            x_t = x[:, t:t+chunk_size]
            y_t, h = self.step(x_t, h)
            ret.append(y_t)
        return torch.cat(ret, 1), h

    def step(self, x, h_t_1):
        fz = fr = fh = x
        if self.apply_input_transformation:
            fz, fr, fh = self.fz(fz), self.fr(fr), self.fh(fh)
        z = self.sigmoid(fz + self.uz(h_t_1))
        r = self.sigmoid(fr + self.ur(h_t_1))
        h_hat = self.act(fh + self.uh(r * h_t_1))
        h = (1 - z) * h_t_1 + z * h_hat
        return h, h


class PerceiverLayer(nn.Module):
    def __init__(self, config, input_size=None):
        super().__init__()
        self.dim = config.hidden_size
        self.input_size = input_size if input_size is not None else self.dim
        self.cross_attn = __ATTENTION_CLASS__(config, mask=True)
        self.cross_rms = RMSNorm(self.input_size, eps=config.rms_norm_eps)
        self.latent_rms = RMSNorm(self.dim, eps=config.rms_norm_eps)
        self.glu_rms = RMSNorm(self.dim, eps=config.rms_norm_eps)
        self.glu = GLU(self.dim, config.intermediate_size, self.dim)
        self.gru = cGRU(self.dim, self.dim, True)
        self.in_proj = nn.Linear(self.input_size, self.dim)
        self.cross_proj = nn.Linear(self.dim, self.dim)
        self.res_proj = nn.Linear(self.dim, self.dim)
        #self.conv1d = CausalConv1d(config, self.dim)
        
        ## LATENT SPACE
        self.num_latent_transformations = 0
        self.latents = nn.ModuleList([
            LatentTransformation(config) for _ in range(self.num_latent_transformations)
        ])
        
        ## OUTPUT SPACE
        self.out_attn = __ATTENTION_CLASS__(config, mask=True)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.out_rms = RMSNorm(self.dim, eps=config.rms_norm_eps)

        self.conv = CausalConv1d(config)

    def forward(
        self, 
        latent_space, 
        input_space, 
        output_space, 
        masks, 
        sA_cache, 
        A_cache_Q_idcs, 
        h_t,
        conv_state
    ):
        mask_2d, mask_4d = masks
        cross_cache, inner_cache, out_cache = sA_cache
        cross_Q_idx, inner_Q_idx, out_Q_idx = A_cache_Q_idcs
        h_1, h_2, h_3 = h_t
        
        residual = latent_space
        x_latent = self.latent_rms(latent_space)
        x_input = self.cross_rms(input_space)
        x_latent = self.cross_proj(x_latent)
        x_input = self.in_proj(x_input)
        latent_space, h_1, cross_cache, cross_Q_idx = self.cross_attn(x_latent, x_input, h_t=h_1, mask=True, A_cache=cross_cache, A_cache_Q_idx=cross_Q_idx) 
        latent_space += x_latent + x_input
        #latent_space += x_input
        
        #latent_space = self.conv1d(latent_space)
        latent_space, conv_state = self.conv(latent_space, conv_state)

        for L in self.latents:
            latent_space, h_2, inner_cache, inner_Q_idx = L(latent_space, h_2, inner_cache, inner_Q_idx)

        #output_space += latent_space

        x_glu = self.glu_rms(latent_space)
        latent_space = self.glu(x_glu) + latent_space
        #y_glu, h_t = self.gru(x_glu)
        #latent_space = latent_space + y_glu

        x_out = self.out_rms(output_space)
        x_out = self.out_proj(x_out)
        output_space, h_3, out_cache, out_Q_idx = self.out_attn(x_out, latent_space, h_t=h_3, mask=False, A_cache=out_cache, A_cache_Q_idx=out_Q_idx) 
        output_space += x_out + x_input
        #output_space = latent_space[:, -output_space.shape[1]:]

        #output_space += self.out_proj(latent_space)
        return (
            latent_space, 
            output_space, 
            (cross_cache, inner_cache, out_cache), 
            (cross_Q_idx, inner_Q_idx, out_Q_idx),
            torch.stack((h_1, h_2, h_3), 0),
            conv_state
        )


class PerceiverEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.input_size = config.input_size
        self.layers = nn.ModuleList([PerceiverLayer(config, self.input_size)])
        for _ in range(self.num_layers-1):
            self.layers.append(PerceiverLayer(config))
        
    def forward(self, chunk_size, *args):
        sA_cache = [([], [], []) for _ in range(self.num_layers)]
        A_cache_Q_idcs = [(0, 0, 0) for _ in range(self.num_layers)]
        #return self.forward_seq(*args, sA_cache=sA_cache, A_cache_Q_idcs=A_cache_Q_idcs)[:2] if self.training else self.forward_autoregressive(*args, sA_cache=sA_cache, A_cache_Q_idcs=A_cache_Q_idcs)
        return self.forward_recurrent(*args, chunk_size=chunk_size)
        
    def forward_seq(
        self, 
        latent_space, 
        input_space, 
        output_space, 
        masks, 
        sA_cache, 
        A_cache_Q_idcs, 
        h_t, 
        conv_state
    ):
        #latent_space = input_space # for testing
        for i, L in enumerate(self.layers):
            latent_space, output_space, sA_cache[i], A_cache_Q_idcs[i], h_t[i], conv_state[i] = L(
                latent_space, input_space, output_space, masks, sA_cache[i], 
                A_cache_Q_idcs[i], h_t[i], conv_state[i])
        return latent_space, output_space, sA_cache, A_cache_Q_idcs, h_t, conv_state

    def forward_recurrent(
        self, 
        latent_space, 
        input_space, 
        output_space, 
        masks, 
        h_t,
        conv_state,
        chunk_size=1
    ):
        B, T, C = input_space.shape
        #latent_space = torch.zeros(B, chunk_size, C).to(input_space.device)
        ret = []
        sA_cache = [([], [], []) for _ in range(self.num_layers)]
        A_cache_Q_idcs = [(0, 0, 0) for _ in range(self.num_layers)]
        for t in range(0, T, chunk_size):
            input_t = input_space[:, t:t+chunk_size]
            latent_space, output_space, sA_cache, A_cache_Q_idcs, h_t, conv_state = self.forward_seq(
                latent_space, input_t, output_space, masks, sA_cache, A_cache_Q_idcs, h_t, conv_state)
            y_t = output_space[:, -chunk_size:]
            #y_t = latent_space[:, -chunk_size:]
            ret.append(y_t)
        out = torch.cat(ret, 1)
        return out, out, h_t, conv_state

    def forward_autoregressive(
        self, 
        latent_space, 
        input_space, 
        output_space, 
        masks, 
        sA_cache, 
        A_cache_Q_idcs,
        h_t
    ):
        B, T, C = output_space.shape
        ret = []
        for t in range(T):
            latent_t = latent_space[:, :t+1]
            #latent_t = torch.zeros(B, t+1, C).to(latent_space.device)
            input_t = input_space[:, :t+1]
            output_t = output_space[:, :t+1]
            masks_t = (None, None)
            y, o, _, _, h_t = self.forward_seq(
                latent_t, 
                input_t, 
                output_t, 
                masks_t, 
                sA_cache=sA_cache, 
                A_cache_Q_idcs=A_cache_Q_idcs,
                h_t=h_t
            )
            y_t = o[:, -1]
            #latent_space[:, t] = y_t
            ret.append(y_t)
        out = torch.stack(ret, 1)
        return out, out, h_t


class PerceiverModel(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dim = config.hidden_size
        #self.latent_size = 4
        self.conv_kernel_size = config.conv_kernel_size
        self.num_layers = config.num_layers
        self.embeddings = PerceiverEmbeddings(config)
        self.model = PerceiverEncoder(config)
        self.post_init()

    def init_latent(self, B, T, l):
        return torch.zeros(B, l, self.dim)
    
    def init_output(self, B, T, l):
        return torch.zeros(B, T, self.dim)

    def init_h(self, B, T, l):
        return torch.zeros(self.num_layers, 3, B, T, T)

    def init_conv_state(self, B):
        return torch.zeros(self.num_layers, B, self.dim, self.conv_kernel_size)

    def prepare_4d_causal_mask(self, mask, seq_len, target_len, dtype, device, B):
        if mask is not None and mask.dim() == 4:
            causal_mask = mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((seq_len, target_len), fill_value=min_dtype, dtype=dtype, device=device)
            if seq_len != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)
            if mask is not None:
                causal_mask = causal_mask.clone()  # copy to contigous memory for in-place edit
                mask_len = mask.shape[-1]
                pad_mask = causal_mask[..., :mask_len] + mask[:, None, None, :]
                pad_mask = pad_mask == 0
                causal_mask[..., :mask_len].masked_fill_(pad_mask, 0)
        return causal_mask

    def _chunk_size(self, T):
        c = -1
        while c <= 0 or T % c != 0:
            c = random.randint(1, T)
        return c

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        input_space = self.embeddings(input_ids, inputs_embeds)
        B, T, C = input_space.shape
        chunk_size = 64 #self._chunk_size(T)
        latent_size = chunk_size
        masks = (
            attention_mask,
            self.prepare_4d_causal_mask(attention_mask, T, T, torch.float, input_space.device, B)
        )
        latent_space = self.init_latent(B, chunk_size, latent_size).to(input_space.device)
        output_space = self.init_output(B, chunk_size, latent_size).to(input_space.device)
        h_t = self.init_h(B, chunk_size, latent_size).to(input_space.device)
        conv_state = self.init_conv_state(B).to(input_space.device)
        latent_space, output_space, h_t, conv_state = self.model(
            chunk_size, 
            latent_space, 
            input_space, 
            output_space, 
            masks,
            h_t,
            conv_state
        )
        #output_space = output_space[:, :T]
        return output_space
    

class PerceiverForCausalLM(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.model = PerceiverModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, inputs_embeds=None, labels=None, num_logits_to_keep=0, **kwargs):
        logits = self.model(input_ids, inputs_embeds)
        logits = self.lm_head(logits[:, -num_logits_to_keep:, :])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fn(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        return PerceiverOutput(
            logits=logits,
            loss=loss
        )


class PerceiverForParityCheck(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = PerceiverModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, inputs_embeds=None, labels=None, **kwargs):
        inputs = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        logits = self.model(inputs_embeds=inputs)
        logits = self.classifier(logits[:, -1])
        loss = self.loss_fn(logits, labels)
        return PerceiverOutput(
            logits=logits,
            loss=loss
        )


class PerceiverForSequenceClassification(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.model = PerceiverModel(config)
        self.pooler = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss() if config.num_labels <= 2 else nn.BCEWithLogitsLoss()
        self.post_init()

    def forward(self, input_ids, inputs_embeds=None, labels=None, **kwargs):
        logits = self.model(input_ids, inputs_embeds)
        logits = self.pooler(logits[:, -1])
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        loss = self.loss_fn(logits, labels)
        return PerceiverOutput(
            logits=logits,
            loss=loss
        )


class PerceiverPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.SiLU()
        self.rms_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.rms_norm(hidden_states)
        return hidden_states


class PerceiverLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = PerceiverPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class PerceiverOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = PerceiverLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class PerceiverForMaskedLM(PerceiverPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.apply_causal_mask = False
        self.model = PerceiverModel(config)
        self.cls = PerceiverOnlyMLMHead(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self,
        input_ids, 
        inputs_embeds=None, 
        attention_mask=None, 
        labels=None, 
        **kwargs
    ):
        logits = self.model(input_ids, inputs_embeds, attention_mask)
        scores = self.cls(logits)
        loss = self.loss_fn(scores.view(-1, self.config.vocab_size), labels.view(-1))
        return PerceiverOutput(
            logits=scores,
            loss=loss
        )
