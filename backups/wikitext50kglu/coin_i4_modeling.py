import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, NamedTuple
import copy
import numpy as np
import random

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


class COINOutput:
    def __init__(self, logits=None, S=None, C=None, loss=None, aux_loss=None):
        self.logits = logits
        self.encoder_hidden_state = None
        self.S = S
        self.C = C
        self.loss = loss
        self.aux_loss = aux_loss


class COINConfig(PretrainedConfig):
    model_type = "ci4"

    def __init__(
        self,
        hidden_size: int = 1024,
        input_size: Optional[int] = None,
        forward_method: str = "chunkwise",
        num_heads: int = 12,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: int = 1536,
        num_layers: int = 1,
        gamma: float = (1 - 1e-6),
        training_chunk_size: Optional[int] = None,
        inference_chunk_size: Optional[int] = 1,
        layer_norm_eps: float = 1e-12,
        rms_norm_eps: float = 1e-05,
        qkv_activation: Optional[str] = "relu",
        group_norm_num: int = 32,
        group_norm_channels: int = 32,
        group_norm_eps: float = 1e-05,
        rope_dim: int = 16,
        apply_decay_mask: bool = False,
        apply_attention_mask: bool = False,
        apply_group_mask: bool = False,
        hidden_dropout_prob: float = 0.1,
        vocab_size: int = 30522,
        num_labels: int = 2,
        max_position_embeddings: int = 512,
        reset_hidden_states: bool = True,
        
        pad_token_id: int = 0,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size if input_size is not None else hidden_size
        self.forward_method = forward_method
        self.num_heads = num_heads
        #self.head_dim = head_dim
        self.num_key_value_heads = num_heads if num_key_value_heads is None else num_key_value_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.gamma = gamma
        self.training_chunk_size = training_chunk_size
        self.inference_chunk_size = inference_chunk_size
        self.layer_norm_eps = layer_norm_eps
        self.rms_norm_eps = rms_norm_eps
        self.qkv_activation = qkv_activation
        self.group_norm_num = group_norm_num
        self.group_norm_channels = group_norm_channels
        self.group_norm_eps = group_norm_eps
        self.rope_dim = rope_dim
        self.apply_decay_mask = apply_decay_mask
        self.apply_attention_mask = apply_attention_mask
        self.apply_group_mask = apply_group_mask
        self.hidden_dropout_prob = hidden_dropout_prob
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.max_position_embeddings = max_position_embeddings
        self.reset_hidden_states = reset_hidden_states
        self.initializer_range = initializer_range


class COINPreTrainedModel(PreTrainedModel):
    config_class = COINConfig
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


class COINEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, inputs_embeds=None):
        X = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        #X = self.norm(X)
        #X = self.dropout(X)
        return X


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fz = nn.Linear(input_size, hidden_size)
        self.uz = nn.Linear(hidden_size, hidden_size)
        self.fr = nn.Linear(input_size, hidden_size)
        self.ur = nn.Linear(hidden_size, hidden_size)
        self.fh = nn.Linear(input_size, hidden_size)
        self.uh = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.SiLU()

    def forward(self, x, h_t_1):
        z = self.sigmoid(self.fz(x) + self.uz(h_t_1))
        r = self.sigmoid(self.fr(x) + self.ur(h_t_1))
        h_hat = self.act(self.fh(x) + self.uh(r * h_t_1))
        h = (1 - z) * h_t_1 + z * h_hat
        return h, h


class GLU(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.size = intermediate_size
        self.fi = nn.Linear(input_size, intermediate_size * 2)
        self.fc = nn.Linear(intermediate_size, output_size)
        self.act = nn.SiLU()

    def forward(self, x):
        z, r = self.fi(x).split([self.size, self.size], -1)
        o = z * self.act(r)
        return self.fc(o)


class MLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.fi = nn.Linear(input_size, intermediate_size)
        self.fc = nn.Linear(intermediate_size, output_size)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.fi(x))
        return self.fc(y)


class RGLU(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, bias=True):
        super().__init__()
        self.fz = nn.Linear(input_size, intermediate_size, bias)
        self.fr = nn.Linear(input_size, intermediate_size, bias)
        self.fc = nn.Linear(intermediate_size, output_size, bias)
        self.fh = nn.Linear(intermediate_size, intermediate_size, bias)
        self.act = nn.SiLU()

    def forward(self, x, h_t_1):
        z = self.fz(x)
        r = self.fr(x)
        o = z * self.act(r)
        o += self.fh(h_t_1)
        y = self.fc(o)
        return y, o


class TranspositionMixer(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.size = intermediate_size
        self.fi = nn.Linear(input_size, intermediate_size * 2)
        self.fc = nn.Linear(intermediate_size, output_size)
        self.act = nn.GELU()
    
    def forward(self, x):
        z, r = self.fi(x).split([self.size, self.size], -1)
        #r = self.act(r.transpose(-2, -1))
        z = (z.transpose(-2, -1)).reshape(z.shape)
        o = self.fc(z)
        return o


class i4Block(nn.Module):
    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.decoder_keys = False
        self.config = config
        self.gamma = config.gamma
        self.apply_decay = config.apply_decay_mask
        self.apply_attention_mask = config.apply_attention_mask
        self.apply_group_mask = config.apply_group_mask
        self.size = output_size

        self.num_heads = config.num_heads
        self.head_dim = getattr(config, "head_dim", self.size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        #print(self.num_heads, self.head_dim, self.num_key_value_heads)
        
        self.W_Q = nn.Parameter(torch.randn(input_size, self.head_dim * self.num_heads) / input_size)
        self.W_K = nn.Parameter(torch.randn(input_size, self.head_dim * self.num_key_value_heads) / input_size)
        self.W_V = nn.Parameter(torch.randn(input_size, self.head_dim * self.num_key_value_heads) / input_size)

        self.b_Q = nn.Parameter(torch.randn(output_size) / output_size)
        
        self.Ub_inner = nn.Linear(output_size, output_size)
        self.Wb_r = nn.Linear(output_size, output_size)
        self.Ub_r = nn.Linear(output_size, output_size)
        self.Wb_z = nn.Linear(output_size, output_size)
        self.Ub_z = nn.Linear(output_size, output_size)

        self.rope = RotaryEmbedding(config.rope_dim)
        self.gate_act = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.group_norm_eps)
        self.fc = nn.Linear(output_size, output_size)

        T = config.max_position_embeddings
        #self.att_bias = nn.Parameter(torch.randn(T, T) / T)
        self.cope = CoPE(T, self.size)
        self.mixer = TranspositionMixer(output_size, config.intermediate_size, output_size)

        # gru parameters
        self.Wb_r = nn.Linear(input_size, output_size)
        self.Ub_r = nn.Linear(output_size, output_size)
        self.Wb_z = nn.Linear(input_size, output_size)
        self.Ub_z = nn.Linear(output_size, output_size)
        self.Wb_h = nn.Linear(input_size, output_size)
        self.Ub_h = nn.Linear(output_size, output_size)

        print("gamma:", self.gamma)

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _qkv(self, x, dx, offset, apply_rope=True, apply_act=True, multi_head=False):
        B, T, C = x.shape
        
        Q = x @ self.W_Q
        if self.decoder_keys and dx is not None:
            K = dx @ self.W_K
            V = dx @ self.W_V
        else:
            K = x @ self.W_K
            V = x @ self.W_V

        if multi_head:
            Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if apply_rope:
            Q = self.rope.rotate_queries_or_keys(Q, offset=offset)
            K = self.rope.rotate_queries_or_keys(K, offset=offset)

        if apply_act and self.act is not None:
            Q = self.act(Q)
            K = self.act(K)
            V = self.act(V)
        return Q, K, V

    def _out(self, x):
        out = self.dropout(x)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = self.mixer(out)
        return out

    def _forward_dbgru(self, x, h_t_1):
        z = F.sigmoid(self.Wb_z(x) + self.Ub_z(h_t_1))
        r = F.sigmoid(self.Wb_r(x) + self.Ub_r(h_t_1))
        h_hat = F.relu(self.Wb_h(x) + self.Ub_h(r * h_t_1))
        h = (1 - z) * h_t_1 + z * h_hat
        return h, h

    def forward_parallel(self, x, dx, h_t_1, group_mask, att_mask, att_mask_4d, offset):
        B, T, C = x.shape
        Q, K, V = self._qkv(x, dx, offset, apply_rope=True, apply_act=True)
        D = _get_D(self.gamma, T).to(x.device)
        if group_mask is not None:
            G = _3d_group_mask(group_mask)

        #Q = Q * group_mask.unsqueeze(-1)
        #K = K * group_mask.unsqueeze(-1)
        
        A = Q @ K.transpose(-2, -1)
        if self.apply_attention_mask:
            #print(att_mask)
            A *= att_mask.view(B, T, 1).repeat(1, 1, T)#.bool()).float()
            #A *= att_mask.view(B, 1, T).repeat(1, T, 1)
        if self.apply_decay:
            A *= D.unsqueeze(0)
        if self.apply_group_mask:
            A *= G
        A @= V
        #A = self._out(A) #+ A
        A = self.fc(A)
        #A = self._out(A)
        #y, h_t = self._forward_dbgru(A, h_t_1)
        #y += A
        y = A

        return y, h_t_1

    def forward_recurrent(self, x, dx, s_t_1, group_mask, att_mask, att_mask_4d, offset):
        B, T, C = x.shape
        Q, K, V = self._qkv(x, dx, offset, apply_rope=True, apply_act=False)
        D = _get_D(self.gamma, T).to(x.device)
        if group_mask is not None:
            G = _3d_group_mask(group_mask)

        s_t = self.gamma * s_t_1 + K.transpose(-2, -1) @ V
        y = Q @ s_t
        y = self.fc(y)
        return y, s_t

    def forward_mixed(self, x, dx, s_t_1, h_t_1, group_mask, att_mask, att_mask_4d, offset):
        if self.training:
            y, h_t = self.forward_parallel(x, dx, h_t_1, group_mask, att_mask, att_mask_4d, offset)
            return y, s_t_1, h_t
        else:
            y, s_t = self.forward_recurrent(x, dx, s_t_1, group_mask, att_mask, att_mask_4d, offset)
            return y, s_t, h_t_1

    def forward_multi_head_parallel(self, x, dx, group_mask, att_mask, att_mask_4d, offset):
        B, T, C = x.shape
        Q, K, V = self._qkv(x, dx, offset, apply_rope=True, apply_act=False, multi_head=True)
        D = _get_D(self.gamma, T).to(x.device)

        K = self._repeat_kv(K, self.num_key_value_groups)
        V = self._repeat_kv(V, self.num_key_value_groups)

        A = Q @ K.transpose(-2, -1)
        if self.apply_attention_mask:
            #print(att_mask)
            #A *= att_mask.view(B, T, 1).repeat(1, 1, T)#.bool()).float()
            A *= att_mask_4d
        if self.apply_decay:
            A *= D.unsqueeze(0)
        
        #print(att_mask_4d)
        A @= V

        A = A.transpose(1, 2).reshape(B, T, self.size)
        #A = self._out(A) + A
        A = self.fc(A)
        return A

    def forward_chunkwise(self, x, dx, s_t_1, h_t_1, group_mask, att_mask, offset, inner_chunk_cache):
        B, T, C = x.shape
        Q, K, V = self._qkv(x, dx, offset, apply_rope=True, apply_act=False)
        D = _get_D(self.gamma, T).to(x.device)

        #Q = Q * group_mask.unsqueeze(-1)
        #K = K * group_mask.unsqueeze(-1)
        #Q = F.softmax(Q, 1)
        #K = F.softmax(K, 1)

        if inner_chunk_cache is None:
            inner_chunk = Q @ K.transpose(-2, -1)
            #inner_chunk /= self.size ** .5
            if self.apply_attention_mask:
                inner_chunk *= att_mask.view(B, T, 1).repeat(1, 1, T)#.bool()).float()
            if self.apply_decay:
                inner_chunk *= D.unsqueeze(0)
            #inner_chunk = inner_chunk.masked_fill_(D.unsqueeze(0) == 0, float("-inf"))
            #inner_chunk = F.softmax(inner_chunk, -1)
            inner_chunk @= V
        else:
            assert inner_chunk_cache.shape == (B, T, self.size), inner_chunk_cache.shape
            inner_chunk = inner_chunk_cache

        #return inner_chunk, s_t_1, h_t_1
        
        #K = K * group_mask.unsqueeze(-1)
        #V = V * group_mask.unsqueeze(-1)

        D_1 = D[-1].view(1, T, 1)
        #s_t = (K.transpose(-2, -1) @ (V * D_1)) + (s_t_1 * (self.gamma ** T))
        s_t = (K.transpose(-2, -1) @ (V)) + (s_t_1 * (self.gamma ** T))
        #if T == 1:
        #    s_t *= group_mask.unsqueeze(-1)

        e = torch.zeros(B, T, 1).to(x.device)
        for i in range(T):
            e[:, i, :] = self.gamma ** (i + 1)
        cross_chunk = (Q @ s_t_1) * e
        #cross_chunk *= group_mask.unsqueeze(-1)

        #inner_chunk = (Q @ s_t)
        
        #r = self.gate_act(self.Wb_r(inner_chunk) + self.Ub_r(h_t_1))
        #z = self.gate_act(self.Wb_z(cross_chunk) + self.Ub_z(h_t_1))

        #h_t = F.relu(inner_chunk + self.Ub_inner(h_t_1)) #* r
        #y_t = h_t + (cross_chunk * z)
        
        h_t = h_t_1
        y_t = inner_chunk + cross_chunk
        
        #y_t = self._out(y_t) + y_t
        y_t = self.fc(y_t)
        
        #o_t, h_t = self._forward_dbgru(y_t, h_t_1)
        #o_t += y_t

        return y_t, s_t, h_t

    def forward_linear_chunkwise(self, x, dx, s_t_1, h_t_1, att_mask, offset, inner_chunk_cache):
        B, T, C = x.shape
        Q, K, V = self._qkv(x, dx, offset, apply_rope=True, apply_act=True)
        D = _get_D(self.gamma, T).to(x.device)

        D_1 = D[-1].view(1, T, 1)
        s_t = (K.transpose(-2, -1) @ (V * D_1)) + (s_t_1 * (self.gamma ** T))

        e = torch.zeros(B, T, 1).to(x.device)
        for i in range(T):
            e[:, i, :] = self.gamma ** (i + 1)
        #cross_chunk = (Q @ s_t_1) * e

        current_chunk = (Q @ s_t)

        h_t = h_t_1
        y_t = current_chunk #+ cross_chunk
        
        y_t = self._out(y_t) + y_t
        return y_t, s_t, h_t

    def chunking_loop(self, x, dx, s_t, h_t, att_mask, offset):
        B, T, C = x.shape
        chunk_size = 16#T# if self.training else 1

        s_t = torch.zeros(B, self.size, self.size).to(x.device)
        h_t = torch.zeros(B, chunk_size, self.size).to(x.device)

        Y = []
        for t in range(0, T, chunk_size):
            x_t = x[:, t:t+chunk_size]
            y_t, s_t, h_t = self.forward_chunkwise(x_t, None, s_t, h_t, None, t + offset)
            Y.append(y_t)
        y = torch.cat(Y, 1)
        #y = self._out(y) + y
        return y, s_t, h_t

    def forward_kv_rep(self, x, dx, s_t_1, h_t_1, att_mask_4d, offset):
        B, T, C = x.shape
        K_1, V_1 = s_t_1
        Q, K, V = self._qkv(x, dx, apply_rope=True, apply_act=False, multi_head=True)
        D = _get_D(self.gamma, T).to(x.device)

        A_inner = Q @ K.transpose(-2, -1)
        A_inner /= self.head_dim
        A_inner = F.softmax(A_inner, -1)
        A_inner @= V

        A_cross = Q @ K_1.transpose(-2, -1)
        A_cross /= self.head_dim ** .5
        A_cross = F.softmax(A_cross, -1)
        A_cross @= V_1

        y = A_inner + A_cross
        y = self.fc(y)

        s_t = [
            K + K_1,
            V + V_1
        ]
        return y, s_t, h_t_1

    def forward_llama(self, x, dx, att_mask_4d, offset):
        B, T, C = x.shape
        Q, K, V = self._qkv(x, None, offset, apply_act=False, multi_head=False)
        D = _get_D(1, T).to(x.device)

        #K = self._repeat_kv(K, self.num_key_value_groups)
        #V = self._repeat_kv(V, self.num_key_value_groups)

        A = (Q @ K.transpose(-2, -1)) / (self.head_dim ** .5)
        #A += att_mask_4d
        #A = A.masked_fill_(D.unsqueeze(0) == 0, float("-inf"))
        A += D.unsqueeze(0)
        A = F.softmax(A, -1)
        #A = F.sigmoid(A)
        #A = F.relu(A**2)
        #A = self.dropout(A)
        #A *= (self.head_dim ** .5)
        A @= V

        #A = A.transpose(1, 2).reshape(B, T, self.size)
        A = self.fc(A)
        return A

    def forward(self, x, dx, s_t_1, h_t_1, group_mask, att_mask, att_mask_4d, offset, forward_method, inner_chunk_cache):
        #return self.forward_recurrent(x, dx, s_t_1, h_t_1, att_mask, offset, inner_chunk_cache)
        
        if forward_method in ("parallel", "inner_chunk_cache"):
            y_t, h_t = self.forward_parallel(x, dx, h_t_1, group_mask, att_mask, att_mask_4d, offset)
            return y_t, s_t_1, h_t
        elif forward_method == "parallel_multi_head":
            y_t = self.forward_multi_head_parallel(x, dx, group_mask, att_mask, att_mask_4d, offset)
            return y_t, s_t_1, h_t_1
        elif forward_method == "chunkwise":
            #return self.forward_linear_chunkwise(x, dx, s_t_1, h_t_1, att_mask, offset, inner_chunk_cache)
            return self.forward_chunkwise(x, dx, s_t_1, h_t_1, group_mask, att_mask, offset, inner_chunk_cache)
        elif forward_method == "llama":
            y_t = self.forward_llama(x, dx, att_mask_4d, offset)
            return y_t, s_t_1, h_t_1
        elif forward_method == "mixed":
            return self.forward_mixed(x, dx, s_t_1, h_t_1, group_mask, att_mask, att_mask_4d, offset)
        else:
            raise ValueError(forward_method)


class COINLayer(nn.Module):
    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.size = output_size
        self.block = i4Block(config, input_size, output_size)
        self.rms = RMSNorm(input_size, eps=config.rms_norm_eps)
        self.post_rms = RMSNorm(input_size, eps=config.rms_norm_eps)
        self.skip_gate = nn.Linear(input_size, output_size * 2)
        self.glu = GLU(output_size, config.intermediate_size, output_size)
        self.rglu = RGLU(input_size, config.intermediate_size, output_size)
        self.mlp = MLP(input_size, config.intermediate_size, output_size)
        self.mixer = TranspositionMixer(output_size, config.intermediate_size, output_size)
        self.gru = GRU(input_size, output_size)

    def forward(self, x, dx, s_t, h_t, group_mask, att_mask, att_mask_4d, offset, forward_method, inner_chunk_cache):
        calc_icc = forward_method == "inner_chunk_cache"
        #if self.training:
        #x = self.mixer(x)
        norm_x = x#self.rms(x) if not calc_icc else x
        #y_t, s_t, h_t = self.block(norm_x, dx, s_t, h_t, group_mask, att_mask, att_mask_4d, offset, forward_method, inner_chunk_cache)
        #y_t = self.mixer(norm_x)
        #y_t = norm_x.transpose(-2, -1).reshape(norm_x.shape)
        #if not calc_icc:
            #y_t = y_t + self.skip_gate(x)
        #    y_t += norm_x
            #z, r = self.skip_gate(x).split([self.size, self.size], -1)
            #r = F.sigmoid(r)
            #y_t = y_t * r + z
        
        #y_t += x#norm_x
        y_t = x
        
        z_t = self.post_rms(y_t)
        #z_t = z_t.transpose(-2, -1).reshape(z_t.shape)
        z_t = self.glu(z_t)
        #z_t = self.mlp(z_t)
        #z_t, h_t = self.rglu(z_t, h_t)
        #z_t, h_t = self.gru(z_t, h_t)
        
        z_t += y_t


        ### revmlp
        #z_t = self.mlp(z_t)
        #z_t = z_t.exp() * y_t
        
        return z_t, dx, s_t, h_t


class COINMultiLayerBlock(nn.Module):
    def __init__(self, config, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([COINLayer(config, input_size, hidden_size)])
        for _ in range(num_layers - 1):
            self.layers.append(COINLayer(config, hidden_size, hidden_size))
        self.num_layers = len(self.layers)
        self.size = hidden_size
        self.forward_method = config.forward_method
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss()

    def context_mask(self, B, T, C, p=0.1):
        out = []
        for _ in range(B):
            ones = torch.ones(T, C)
            for _ in range(int(T * p)):
                pos = random.randint(0, T-1)
                while (ones[pos, :].all() == 0).item():
                    pos = random.randint(0, T-1)
                ones[pos, :] = 0
            out.append(ones)
        return torch.stack(out)

    def forward(self, x, dx, s_t, h_t, group_mask, att_mask, att_mask_4d, offset, inner_chunk_cache):
        if inner_chunk_cache is None:
            inner_chunk_cache = [None for _ in range(self.num_layers)]
        assert self.num_layers == s_t.shape[0] == h_t.shape[0], f"{self.num_layers} != {s_t.shape}[0] != {h_t.shape}[0]"
        forward_method = self.forward_method
        s_out = []
        h_out = []
        aux_loss = torch.Tensor([0])[0].to(x.device)
        for L, s_t_n, h_t_n, icc in zip(self.layers, s_t, h_t, inner_chunk_cache):
            #B, T, C = x.shape
            #masked_x = torch.cat((
            #    x,
            #    x #* self.context_mask(B, T, C, p=0.2).to(x.device)
            #), 0)
            
            x, dx, s_t_n, h_t_n = L(x, dx, s_t_n, h_t_n, group_mask, att_mask, att_mask_4d, offset, forward_method, icc)
            #x_res, dx, s_t_n, h_t_n = L(masked_x, dx, s_t_n, h_t_n, att_mask, offset, forward_method, icc)
            #aux_loss += self.loss_fn(x_res[B:][..., :-1, :].reshape(-1, self.size), x.argmax(-1)[..., 1:].reshape(-1)) / self.num_layers
            #x = x_res[:B]

            s_out.append(s_t_n)
            h_out.append(h_t_n)
        s_t = torch.stack(s_out)
        h_t = torch.stack(h_out)
        return x, dx, s_t, h_t, aux_loss

    def calc_inner_chunk(self, x, dx, att_mask, offset):
        cache = []
        for L in self.layers:
            x, dx, _, _ = L(x, dx, None, None, att_mask, offset, "inner_chunk_cache", None)
            cache.append(x)
        return torch.stack(cache)


class COINSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layers = COINMultiLayerBlock(config, config.input_size, config.hidden_size, config.num_layers)
        self.num_layers = config.num_layers

    def init_h(self, B, chunk_size, device):
        return torch.zeros(self.num_layers, B, chunk_size, self.hidden_size).to(device)
        #return torch.zeros(self.num_layers, B, chunk_size, self.config.intermediate_size).to(device)

    def init_s(self, B, chunk_size, device):
        return torch.zeros(self.num_layers, B, self.hidden_size, self.hidden_size).to(device)

    def init_hidden_state(self, B, chunk_size, device) -> None:
        return (
            self.init_s(B, chunk_size, device),
            self.init_h(B, chunk_size, device)
        )

    def forward(
        self, 
        encoder_query, 
        decoder_query=None, 
        S=None, 
        group_mask=None, 
        attention_mask=None, 
        attention_mask_4d=None,
        offset=0, 
        override_inference=False
    ):
        B, T, C = encoder_query.shape

        if override_inference or not self.training:
            chunk_size = self.config.inference_chunk_size
        else:
            chunk_size = self.config.training_chunk_size
        if chunk_size is None:
            chunk_size = T

        if S is None:
            s_t, h_t = self.init_s(B, chunk_size, encoder_query.device), self.init_h(B, chunk_size, encoder_query.device)
        else:
            s_t, h_t = S
        if h_t.shape[1] != T:
            h_t = self.init_h(B, chunk_size, encoder_query.device)
        if self.config.reset_hidden_states:
            s_t *= 0
            h_t *= 0

        inner_chunk_cache = None
        #if self.training:
        #inner_chunk_cache = self.layers.calc_inner_chunk(encoder_query, decoder_query, attention_mask, offset)
        
        Y = []
        dY = []
        aux_loss = torch.Tensor([0])[0].to(encoder_query.device)
        for t in range(0, T, chunk_size):
            eq = encoder_query[:, t:t+chunk_size]
            dq = decoder_query[:, t:t+chunk_size] if decoder_query is not None else None
            gm = group_mask[:, t:t+chunk_size] if group_mask is not None else None
            am = attention_mask[:, t:t+chunk_size] if attention_mask is not None else None
            am4d = attention_mask_4d[:, :, t:t+chunk_size] if attention_mask_4d is not None else None
            icc = inner_chunk_cache[:, :, t:t+chunk_size] if inner_chunk_cache is not None else None
            
            #y_t, dy_t, s_t, h_t = self.time_step(eq, dq, s_t, h_t, am, offset + t)
            y_t, dy_t, s_t, h_t, aul = self.layers(eq, dq, s_t, h_t, gm, am, am4d, offset + t, icc)
            aux_loss += aul

            Y.append(y_t)
            dY.append(dy_t)
        
        out = torch.cat(Y, 1) #+ encoder_query
        return out, [s_t, h_t], aux_loss


class COINPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

    def forward(self, X):
        X = X[..., -1, :]
        X = self.dense(X)
        X = self.act(X)
        return X


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    prepare_for_softmax: bool
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        if prepare_for_softmax:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        else:
            causal_mask = torch.ones((sequence_length, target_length), dtype=dtype, device=device)
        if sequence_length != 1:
            if prepare_for_softmax:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask = torch.triu(causal_mask, diagonal=0).transpose(-2, -1)
        #causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, 0
            )

    return causal_mask


class COINModel(COINPreTrainedModel):
    def __init__(self, config, apply_embeddings=True):
        super().__init__(config)
        self.forward_method = config.forward_method
        self.apply_embeddings = apply_embeddings
        if self.apply_embeddings:
            self.encoder_embeddings = COINEmbeddings(config)
            self.decoder_embeddings = COINEmbeddings(config)
        self.sampler = COINSampler(config)
        self.pooler = COINPooler(config)
        self.post_init()

    def forward(
        self, 
        inputs, 
        decoder_inputs=None, 
        S=None, 
        C=None, 
        inputs_embeds=None, 
        decoder_inputs_embeds=None, 
        group_mask=None,
        attention_mask=None, 
        offset=0, 
        override_inference=False
    ):
        B, T = inputs.shape[:2]
        attention_mask_4d = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            T,
            T,
            torch.float,
            inputs.device,
            B,
            self.forward_method in ("llama", "kv_rep")
        )
        
        if self.apply_embeddings:
            inputs = self.encoder_embeddings(inputs, inputs_embeds)
            decoder_inputs = self.decoder_embeddings(decoder_inputs, decoder_inputs_embeds) \
                if decoder_inputs is not None or decoder_inputs_embeds is not None else None
        y, S, aux_loss = self.sampler(
            inputs, 
            decoder_inputs, 
            S, 
            group_mask, 
            attention_mask, 
            attention_mask_4d,
            offset, 
            override_inference
        )
        pooled_y = self.pooler(y)
        return y, pooled_y, S, aux_loss


class COINForSequenceClassification(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.coin = COINModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, S, C, inputs_embeds=None, decoder_inputs_embeds=None, group_mask=None, attention_mask=None, offset=0, labels=None, override_inference=False, **kwargs):
        out, pooled_out, S, aux_loss = self.coin(
            input_ids, 
            decoder_input_ids, 
            S, 
            C, 
            inputs_embeds, 
            decoder_inputs_embeds, 
            group_mask,
            attention_mask, 
            offset, 
            override_inference
        )
        logits = self.dropout(pooled_out)
        logits = self.classifier(logits)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return COINOutput(
            logits=logits,
            S=S,
            C=C,
            loss=loss
        )


class COINForParityCheck(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.coin = COINModel(config, apply_embeddings=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids=None, S=None, C=None, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, override_inference=False, **kwargs):
        inputs = F.one_hot(input_ids, self.config.vocab_size).float()
        decoder_inputs = F.one_hot(decoder_input_ids, self.config.vocab_size).float() if decoder_input_ids is not None else None
        out, pooled_out, S = self.coin(inputs, decoder_inputs, S, C, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, override_inference)
        logits = out[:, -1]
        logits = self.classifier(logits)
        loss = self.loss_fn(logits, labels)
        return COINOutput(
            logits=logits,
            S=S,
            C=C,
            loss=loss
        )


class COINForCausalLM(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self, 
        input_ids, 
        decoder_input_ids=None, 
        S=None, 
        C=None, 
        inputs_embeds=None, 
        decoder_inputs_embeds=None, 
        group_mask=None,
        attention_mask=None, 
        offset=0, 
        labels=None, 
        override_inference=False,
        num_logits_to_keep=0,
        **kwargs
    ):
        logits, _, S, aux_loss = self.coin(
            input_ids, 
            decoder_input_ids, 
            S, 
            C, 
            inputs_embeds, 
            decoder_inputs_embeds, 
            group_mask,
            attention_mask, 
            offset, 
            override_inference
        )
        logits = self.lm_head(logits[:, -num_logits_to_keep:, :])
        # shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        if self.training:
            loss = self.loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss += aux_loss
        else:
            loss = self.loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        #if aux_loss is not None:
        #    loss += aux_loss
        
        return COINOutput(
            logits=logits,
            S=S,
            C=C,
            loss=loss,
            aux_loss=aux_loss
        )


class COINPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_act_func(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        x = self.dense(x)
        x = self.transform_act_fn(x)
        x = self.layer_norm(x)
        return x


class COINLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = COINPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        # former self.decoder.biad

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x


class COINForMaskedLM(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.coni = COINModel(config)
        self.cls = COINLMPredictionHead(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self, 
        input_ids, 
        decoder_input_ids=None, 
        S=None, 
        C=None, 
        inputs_embeds=None, 
        decoder_inputs_embeds=None, 
        group_mask=None,
        attention_mask=None, 
        offset=0, 
        labels=None, 
        override_inference=False,
        **kwargs
    ):
        logits, _, S, aux_loss = self.coin(
            input_ids, 
            decoder_input_ids, 
            S, 
            C, 
            inputs_embeds, 
            decoder_inputs_embeds, 
            group_mask,
            attention_mask, 
            offset, 
            override_inference
        )
        logits = self.cls(logits)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return COINOutput(
            logits=logits,
            S=S,
            C=C,
            loss=loss,
            aux_loss=aux_loss
        )
