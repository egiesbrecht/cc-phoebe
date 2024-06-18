import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import copy
import numpy as np

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


def str_tensor_list_shape(X, prefix=""):
    if X is None:
        return f"{prefix}None"
    elif isinstance(X, torch.Tensor):
        return f"{prefix}Tensor {X.shape}"
    elif isinstance(X, list):
        return "\n".join([f"{prefix}List [\n{str_tensor_list_shape(n, f"{prefix}    ")}\n{prefix}]" for n in X])


def _create_decay_mask(config, schema):
    lower_limit = 32
    upper_limit = 512#config.hidden_size  #768  #512

    if config.reverse_decay:
        lower_limit, upper_limit = upper_limit, lower_limit

    #if heads is None:
    #    heads = config.num_hidden_layers
    heads = len(schema) + schema.count(1)

    gamma_parts = config.num_decay_parts
    ceil_head = math.ceil(heads / gamma_parts)
    floor_head = math.floor(heads / gamma_parts)
    gammas = (1 - torch.cat(
        (torch.exp(torch.linspace(math.log(1 / lower_limit), math.log(1 / upper_limit), ceil_head)),) * (gamma_parts - 1) +
        (torch.exp(torch.linspace(math.log(1 / lower_limit), math.log(1 / upper_limit), floor_head)),)
        , dim=-1)).detach().cpu().tolist()
    ret = []
    i = 0
    for f in schema:
        if f:
            ret.append([gammas[i], gammas[i+1]])
            i += 2
        else:
            ret.append(gammas[i])
            i += 1
    return ret


def _create_encoder_decoder_decay_mask(config, schema):
    lower_limit = 32
    upper_limit = config.hidden_size
    if config.reverse_decay:
        lower_limit, upper_limit = upper_limit, lower_limit
    num_enc = schema.count(0)
    num_dec = schema.count(1) * 2
    fdv = config.fixed_decay_value
    if fdv is not None:
        enc_gammas = [fdv for _ in range(num_enc)]
        dec_gammas = [fdv for _ in range(num_dec)]
    else:
        enc_gammas = (1 - torch.exp(torch.linspace(math.log(1 / lower_limit), math.log(1 / upper_limit), num_enc))).detach().cpu().tolist()
        dec_gammas = (1 - torch.exp(torch.linspace(math.log(1 / lower_limit), math.log(1 / upper_limit), num_dec))).detach().cpu().tolist()
    i_enc, i_dec = 0, 0
    gammas = []
    for f in schema:
        if f:
            gammas.append([dec_gammas[i_dec], dec_gammas[i_dec+1]])
            i_dec += 2
        else:
            gammas.append(enc_gammas[i_enc])
            i_enc += 1
    return gammas


RSIZE = 32
HSIZE = 32


class COINConfig(PretrainedConfig):
    model_type = "Consecutive Chain-Of-Input Network"

    def __init__(
            self,
            num_heads: int = RSIZE,
            hidden_size: int = HSIZE * RSIZE,
            fixed_intermediate_size: bool = False,
            intermediate_size: int = HSIZE * RSIZE * 1,
            intermediate_factor: Union[int, float] = 1,
            hidden_act: str = "tanh",
            hidden_retention_act: str = "relu",
            hidden_out_act: str = "relu",
            layer_norm_eps: float = 1e-12,
            retention_group_norm_eps: float = 1e-05,
            num_hidden_layers: int = 2,
            group_norm_num: Optional[int] = None,
            group_norm_channels: Optional[int] = None,
            hidden_dropout_prob: float = 0.1,
            max_position_embeddings: int = 512,
            vocab_size: int = 30_522,
            num_labels: int = 2,
            pad_token_id: int = 0,
            decoder_start_token_id: int = 0,
            initializer_range: float = 0.02,
            reverse_decay: bool = False,
            num_decay_parts: int = 1,
            apply_decay: bool = False,
            fixed_ffn_intermediate_size: bool = False,
            ffn_intermediate_size: int = HSIZE * RSIZE * 4,
            ffn_intermediate_factor: Union[int, float] = 4,
            apply_ffn: bool = True,
            forward_method: str = "parallel",
            apply_decoder_heads: bool = True,
            num_regions: int = 2,
            share_S: bool = False,  # share S between layers
            decoder_output: str = "adaptive",  # adaptive, strict or none
            decoder_schema: List[bool] = [0, 1],
            cross_encoder_schema: List[bool] = [1, 0],
            apply_softmax_gate: bool = True,
            revert_decoder: bool = True,
            disable_teacher_forcing: bool = False,
            allow_encoder_teacher_forcing: bool = False,
            num_repetitions: int = 1,
            apply_hidden_pos_offset: bool = False,
            local_recurrence_check: bool = False,
            num_local_chunks: int = 4,
            global_recurrence_check: bool = False,
            num_global_chunks: int = 4,
            fixed_decay_value: Optional[float] = None,
            reset_S_n_state: bool = False,
            rms_norm_eps: float = 1e-8,
            rope_dim: int = 16,
            block_io_schema: Optional[List[int]] = None,
            experts_schema: Optional[List[int]] = None,
            switch_ii_decoder_ii: bool = False,
            num_query_heads: int = 2,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.fixed_intermediate_size = fixed_intermediate_size
        self.intermediate_factor = intermediate_factor
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_retention_act = hidden_retention_act
        self.hidden_out_act = hidden_out_act
        self.layer_norm_eps = layer_norm_eps
        self.retention_group_norm_eps = retention_group_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.group_norm_num = group_norm_num if group_norm_num is not None else num_heads
        self.group_norm_channels = group_norm_channels if group_norm_channels is not None else hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.initializer_range = initializer_range
        self.reverse_decay = reverse_decay
        self.num_decay_parts = num_decay_parts
        self.apply_decay = apply_decay
        self.fixed_ffn_intermediate_size = fixed_ffn_intermediate_size
        self.ffn_intermediate_size = ffn_intermediate_size
        self.ffn_intermediate_factor = ffn_intermediate_factor
        self.apply_ffn = apply_ffn
        assert forward_method in ("parallel", "chunkwise")
        self.forward_method = forward_method
        self.apply_decoder_heads = apply_decoder_heads
        self.num_regions = num_regions
        self.share_S = share_S
        assert decoder_output in ("adaptive", "strict", "none")
        self.decoder_output = decoder_output
        self.decoder_schema = decoder_schema
        self.cross_encoder_schema = cross_encoder_schema
        self.apply_softmax_gate = apply_softmax_gate
        self.revert_decoder = revert_decoder
        self.disable_teacher_forcing = disable_teacher_forcing
        self.allow_encoder_teacher_forcing = allow_encoder_teacher_forcing
        self.num_repetitions = num_repetitions
        self.apply_hidden_pos_offset = apply_hidden_pos_offset
        self.local_recurrence_check = local_recurrence_check
        self.num_local_chunks = num_local_chunks
        self.global_recurrence_check = global_recurrence_check
        self.num_global_chunks = num_global_chunks
        self.fixed_decay_value = fixed_decay_value
        self.reset_S_n_state = reset_S_n_state
        self.rms_norm_eps = rms_norm_eps
        self.rope_dim = rope_dim
        self.block_io_schema = block_io_schema
        self.experts_schema = experts_schema
        self.switch_ii_decoder_ii = switch_ii_decoder_ii
        self.num_query_heads = num_query_heads


class COINOutputClass:
    def __init__(self, logits=None, encoder_hidden_state=None, S=None, C=None, loss=None, aux_loss=None):
        self.logits = logits
        self.encoder_hidden_state = encoder_hidden_state
        self.S = S
        self.C = C
        self.loss = loss
        self.aux_loss = aux_loss


class COINPreTrainedModel(PreTrainedModel):
    config_class = COINConfig
    base_model_prefix = "COIN"
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
        E = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds

        #emb_f = self.filter_embeddings(input_ids)
        #E *= F.softmax(emb_f, dim=1)
        X = self.norm(E)
        X = self.dropout(X)
        return X


class SingleHeadQKV(nn.Module):
    def __init__(self, config, gamma, in_size, inter_size, out_size, apply_decay_mask=None, apply_attention_mask=False):
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.apply_attention_mask = apply_attention_mask
        self.apply_decay = apply_decay_mask if apply_decay_mask is not None else config.apply_decay
        self.in_size = in_size
        self.inter_size = inter_size
        self.out_size = out_size
        self.W_Q = nn.Parameter(torch.randn(self.in_size, self.inter_size) / self.in_size)
        self.W_K = nn.Parameter(torch.randn(self.in_size, self.inter_size) / self.in_size)
        self.W_V = nn.Parameter(torch.randn(self.in_size, self.out_size) / self.in_size)
        self.rope = RotaryEmbedding(config.rope_dim)
        self.xpos = XPOS(self.inter_size)
        self.act = get_act_func(config.hidden_retention_act)
        self.out_act = get_act_func(config.hidden_out_act)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        T = config.max_position_embeddings
        #self.att_bias = nn.Parameter(torch.randn(T, T) / T)
        self.cope = CoPE(T, self.inter_size)

    def _qkv(self, X_Q, X_KV, offset):
        if X_KV is None:
            K = X_Q @ self.W_K
            V = X_Q @ self.W_V
        else:
            K = X_KV @ self.W_K
            V = X_KV @ self.W_V
        Q = X_Q @ self.W_Q

        Q = self.rope.rotate_queries_or_keys(Q, offset=offset)
        K = self.rope.rotate_queries_or_keys(K, offset=offset)

        if self.act is not None:
            Q = self.act(Q)
            K = self.act(K)
            V = self.act(V)
        return Q, K, V

    def _out(self, X_Q, out):
        B, T = out.shape[:2]
        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        out = out.transpose(-2, -1).contiguous().view(B, T, self.out_size)
        return out

    def forward_parallel(self, X_Q, X_KV=None, att_mask=None, offset=0):
        B, T, C = X_Q.shape
        Q, K, V = self._qkv(X_Q, X_KV, offset)
        A = Q @ K.transpose(-2, -1)
        if self.apply_attention_mask:
            if att_mask is None:
                att_mask = torch.ones(B, T).to(X_Q.device).triu()#.transpose(-2, -1)
            #A *= att_mask.view(B, 1, T).repeat(1, T, 1)
            A *= att_mask.view(B, T, 1).repeat(1, 1, T).transpose(-2, -1)
        if self.apply_decay:
            D = _get_D(self.gamma, T).unsqueeze(0).to(X_Q.device)
            A *= D#.transpose(-2, -1)
        
        #A = F.softmax(A, -1)
        #A += self.cope(Q, A)

        #A += self.att_bias
        out = A @ V
        out = self._out(X_Q, out) 
        return out

    def forward_chunkwise(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        B, T, C = X_Q.shape
        Q, K, V = self._qkv(X_Q, X_KV, offset)
        D = _get_D(self.gamma, T).to(X_Q.device)
        
        inner_chunk = Q @ K.transpose(-2, -1)
        if self.apply_attention_mask:
            inner_chunk *= att_mask.view(B, T, 1).repeat(1, 1, T)#.bool()).float()
        if self.apply_decay:
            inner_chunk *= D.unsqueeze(0)
        inner_chunk @= V

        R_i = (K.transpose(-2, -1) @ (V * D[-1].view(1, T, 1))) + (S_n * (self.gamma ** T))
        e = torch.zeros(B, T, 1).to(X_Q.device)
        for i in range(T):
            e[:, i, :] = self.gamma ** (i + 1)
        cross_chunk = (Q @ S_n) * e

        out = inner_chunk + cross_chunk
        out = self._out(X_Q, out)
        return out, R_i

    def forward(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        if self.config.forward_method == "parallel":
            y = self.forward_parallel(X_Q, X_KV, att_mask, offset)
            return y, S_n
        elif self.config.forward_method == "chunkwise":
            return self.forward_chunkwise(X_Q, X_KV, S_n, att_mask, offset)
        else:
            raise ValueError(self.config.forward_method)
    

class MultiHeadQKV(nn.Module):
    def __init__(self, config, gamma, in_size, inter_size, out_size, apply_decay_mask=None, apply_attention_mask=False):
        super().__init__()
        self.config = config
        self.num_heads = 8
        assert in_size % self.num_heads == 0
        self.head_inp_size = in_size // self.num_heads
        assert out_size % self.num_heads == 0
        self.head_out_size = out_size // self.num_heads
        assert inter_size % self.num_heads == 0
        self.head_inter_size = inter_size // self.num_heads
        #self.head_size = in_size // self.num_heads
        #print(self.head_inp_size, self.head_inter_size, self.head_out_size)
        self.heads = nn.ModuleList([
            SingleHeadQKV(config, gamma, self.head_inp_size, self.head_inter_size, self.head_out_size, apply_decay_mask, apply_attention_mask)
            for _ in range(self.num_heads)
        ])
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj_out = nn.Linear(out_size, out_size)
    
    def forward(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        #Q_head_chunks = X_Q.split(self.head_inp_size, -1)
        #KV_head_chunks = X_KV.split(self.head_inp_size, -1) if X_KV is not None else [None for _ in range(self.num_heads)]
        Q_head_chunks = X_Q.chunk(self.num_heads, -1)
        KV_head_chunks = X_KV.chunk(self.num_heads, -1) if X_KV is not None else [None for _ in range(self.num_heads)]
        y_out = []
        assert len(Q_head_chunks) == len(KV_head_chunks) == len(self.heads), f"{len(Q_head_chunks)} != {len(KV_head_chunks)} != {len(self.heads)}"
        for H, q, kv in zip(self.heads, Q_head_chunks, KV_head_chunks):
            y, S_n = H(q, kv, S_n, att_mask, offset)
            y_out.append(y)
        out = torch.cat(y_out, -1)
        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        out = self.proj_out(out)
        return out, S_n


class MultiQueryQKV(nn.Module):
    def __init__(self, config, gamma, in_size, inter_size, out_size, apply_decay_mask=None, apply_attention_mask=False):
        super().__init__()
        self.num_query_head = config.num_query_heads
        self.config = config
        self.gamma = gamma
        self.apply_attention_mask = apply_attention_mask
        self.apply_decay = apply_decay_mask if apply_decay_mask is not None else config.apply_decay
        self.in_size = in_size
        self.inter_size = inter_size
        self.out_size = out_size
        self.W_Q = nn.Parameter(torch.randn(self.num_query_head, self.in_size, self.inter_size) / self.in_size)
        self.W_K = nn.Parameter(torch.randn(self.in_size, self.inter_size) / self.in_size)
        self.W_V = nn.Parameter(torch.randn(self.in_size, self.out_size) / self.in_size)
        self.rope = RotaryEmbedding(config.rope_dim)
        self.xpos = XPOS(self.inter_size)
        self.act = get_act_func(config.hidden_retention_act)
        self.out_act = get_act_func(config.hidden_out_act)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj_G = nn.Linear(in_size, out_size)
        self.cope = CoPE(config.max_position_embeddings, self.inter_size)

    def _qkv(self, X_Q, X_KV, offset):
        if X_KV is None:
            K = X_Q @ self.W_K
            V = X_Q @ self.W_V
        else:
            K = X_KV @ self.W_K
            V = X_KV @ self.W_V

        #Q = X_Q @ self.W_Q
        Qs = torch.einsum("btc, nco -> nbto", X_Q, self.W_Q)

        Qs = self.rope.rotate_queries_or_keys(Qs, offset=offset)
        K = self.rope.rotate_queries_or_keys(K, offset=offset)

        if self.act is not None:
            Qs = self.act(Qs)
            K = self.act(K)
            V = self.act(V)
        return Qs, K, V

    def _out(self, X_Q, out):
        B, T = out.shape[:2]
        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        out = out.transpose(-2, -1).contiguous().view(*out.shape[:-1], self.out_size)
        return out

    def forward_parallel(self, X_Q, X_KV=None, att_mask=None, offset=0):
        B, T, C = X_Q.shape
        #U = self.proj_G(X_Q)
        Qs, K, V = self._qkv(X_Q, X_KV, offset)
        #K = (1 / self.inter_size)**.5 * K
        #V = (1 / self.inter_size)**.5 * V
        #A = Q @ K.transpose(-2, -1)

        As = torch.einsum("nbtc, bco -> nbto", Qs, K.transpose(-2, -1))
        #A = As.sum(0)

        if self.apply_attention_mask:
            #A *= att_mask.view(B, 1, T).repeat(1, T, 1)
            As *= att_mask.view(1, B, T, 1).repeat(1, 1, 1, T)
        if self.apply_decay:
            D = _get_D(self.gamma, T).view(1, 1, T, T).to(X_Q.device)
            As *= D#.transpose(-2, -1)

        #As = (As + self.cope(Qs, As)) #/ self.inter_size ** .5

        As = As.sum(0)
        #As *= F.sigmoid(self.proj_G(X_Q))
        #As = As.transpose(-2, -1)
        #A += self.att_bias
        #As = As.sum(0)
        out = As @ V
        #out = torch.einsum("nbtc, bto -> nbto", As, V)
        #out *= U
        #out *= F.softmax(self.proj_G(X_Q), 0)
        #out = out.sum(0)
        out = self._out(X_Q, out)#.sum(0)
        return out

    def forward_chunkwise(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        B, T, C = X_Q.shape
        Qs, K, V = self._qkv(X_Q, X_KV, offset)
        D = _get_D(self.gamma, T).to(X_Q.device)
        
        #inner_chunk = Q @ K.transpose(-2, -1)
        inner_chunk = torch.einsum("nbtc, bco -> nbto", Qs, K.transpose(-2, -1))
        if self.apply_attention_mask:
            inner_chunk *= att_mask.view(B, T, 1).repeat(1, 1, T)#.bool()).float()
        if self.apply_decay:
            inner_chunk *= D.unsqueeze(0)
        inner_chunk = (inner_chunk @ V).sum(0)

        R_i = (K.transpose(-2, -1) @ (V * D[-1].view(1, T, 1))) + (S_n * (self.gamma ** T))
        e = torch.zeros(B, T, 1).to(X_Q.device)
        for i in range(T):
            e[..., i, :] = self.gamma ** (i + 1)
        #cross_chunk = (Q @ S_n) * e
        cross_chunk = torch.einsum("nbtc, bco -> nbto", Qs, S_n) * e
        cross_chunk = cross_chunk.sum(0)

        out = (inner_chunk + cross_chunk)#.sum(0)
        out = self._out(X_Q, out)
        return out, R_i

    def forward(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        if self.config.forward_method == "parallel":
            y = self.forward_parallel(X_Q, X_KV, att_mask, offset)
            return y, S_n
        elif self.config.forward_method == "chunkwise":
            return self.forward_chunkwise(X_Q, X_KV, S_n, att_mask, offset)
        else:
            raise ValueError(self.config.forward_method)
    

class MHeadQKV(nn.Module):
    def __init__(self, config, gamma, in_size, inter_size, out_size, apply_decay_mask=None, apply_attention_mask=False):
        super().__init__()
        self.num_query_heads = config.num_query_heads
        self.config = config
        self.gamma = gamma
        self.apply_attention_mask = apply_attention_mask
        self.apply_decay = apply_decay_mask if apply_decay_mask is not None else config.apply_decay
        self.in_size = in_size
        self.inter_size = inter_size
        self.out_size = out_size
        self.W_Q = nn.Parameter(torch.randn(self.in_size, self.inter_size * self.num_query_heads) / self.in_size)
        self.W_K = nn.Parameter(torch.randn(self.in_size, self.inter_size * self.num_query_heads) / self.in_size)
        self.W_V = nn.Parameter(torch.randn(self.in_size, self.out_size * self.num_query_heads) / self.in_size)
        self.rope = RotaryEmbedding(config.rope_dim)
        self.xpos = XPOS(self.inter_size)
        self.act = get_act_func(config.hidden_retention_act)
        self.out_act = get_act_func(config.hidden_out_act)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.skip_at = SkipAt(inter_size, config.max_position_embeddings)

    def _to_scores(self, X, W):
        B, T, C = X.shape
        o = X @ W
        return o.view(B, T, self.num_query_heads, C).transpose(1, 2)

    def _qkv(self, X_Q, X_KV, offset):
        if X_KV is None:
            #K = X_Q @ self.W_K
            #V = X_Q @ self.W_V
            K = self._to_scores(X_Q, self.W_K)
            V = self._to_scores(X_Q, self.W_V)
        else:
            #K = X_KV @ self.W_K
            #V = X_KV @ self.W_V
            K = self._to_scores(X_KV, self.W_K)
            V = self._to_scores(X_KV, self.W_V)

        #Q = X_Q @ self.W_Q
        #Qs = torch.einsum("btc, nco -> nbto", X_Q, self.W_Q)
        Q = self._to_scores(X_Q, self.W_Q)

        Q = self.rope.rotate_queries_or_keys(Q, offset=offset)
        K = self.rope.rotate_queries_or_keys(K, offset=offset)

        if self.act is not None:
            Q = self.act(Q)
            K = self.act(K)
            V = self.act(V)
        return Q, K, V

    def _out(self, X_Q, out):
        B, T = out.shape[:2]
        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(*out.shape[:-1], self.out_size)
        return out

    def forward_parallel(self, X_Q, X_KV=None, att_mask=None, offset=0):
        B, T, C = X_Q.shape
        Q, K, V = self._qkv(X_Q, X_KV, offset)
        D = _get_D(self.gamma, T).to(X_Q.device)
        A = Q @ K.transpose(-2, -1)
        if self.apply_attention_mask:
            #A *= att_mask.view(B, 1, T).repeat(1, T, 1)
            A *= att_mask.view(B, 1, T, 1).repeat(1, 1, 1, T)
        if self.apply_decay:
            D = _get_D(self.gamma, T).view(1, 1, T, T).to(X_Q.device)
            A *= D#.transpose(-2, -1)
        A = A.transpose(-2, -1)
        out = A @ V

        #out = self.skip_at(out)

        out = out.sum(1)
        out = self._out(X_Q, out)
        
        #out = self.skip_at(out)
        
        return out

    def forward(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        if self.config.forward_method == "parallel":
            y = self.forward_parallel(X_Q, X_KV, att_mask, offset)
            return y, S_n
        elif self.config.forward_method == "chunkwise":
            return self.forward_chunkwise(X_Q, X_KV, S_n, att_mask, offset)
        else:
            raise ValueError(self.config.forward_method)



class i2D_QKV(nn.Module):
    def __init__(self, config, gamma, in_size, inter_size, out_size, apply_decay_mask=None, apply_attention_mask=False):
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.in_size = in_size
        self.out_size = out_size
        self.inter_size = in_size
        self.apply_decay = apply_decay_mask if apply_decay_mask is not None else config.apply_decay
        self.W_Q = nn.Parameter(torch.randn(self.in_size, self.inter_size) / self.in_size)
        self.W_K = nn.Parameter(torch.randn(self.in_size, self.inter_size) / self.in_size)
        self.W_V = nn.Parameter(torch.randn(self.in_size, self.inter_size) / self.in_size)
        self.W_proj = nn.Parameter(torch.randn(self.inter_size, self.out_size) / self.inter_size)
        self.proj_bias = nn.Parameter(torch.randn(self.out_size) / self.out_size)
        self.rope = RotaryEmbedding(config.rope_dim)
        self.xpos = XPOS(self.inter_size)
        self.act = get_act_func(config.hidden_retention_act)
        self.out_act = get_act_func(config.hidden_out_act)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.layer_norm = nn.LayerNorm(self.out_size, eps=config.layer_norm_eps)
        self.rms_norm = RMSNorm(self.out_size, eps=config.rms_norm_eps)                                   
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _qkv(self, X_Q, X_K, X_V, offset):
        Q = X_Q @ self.W_Q
        K = (X_Q @ self.W_K) if X_K is None else (X_K @ self.W_K)
        V = (X_Q @ self.W_V) if X_V is None else (X_V @ self.W_V)
        
        Q = self.rope.rotate_queries_or_keys(Q, offset=offset)
        K = self.rope.rotate_queries_or_keys(K, offset=offset)
        #Q = self.xpos(Q)
        #K = self.xpos(K, downscale=True)

        if self.act is not None:
            Q = self.act(Q)
            K = self.act(K)
            V = self.act(V)
        return Q, K, V
        
    def _out(self, X_Q, out):
        B, T = out.shape[:2]
        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(B, T, self.out_size)
        return out

    def forward_parallel(self, X_Q, X_K=None, X_V=None, att_mask=None, offset=0):
        B, T, C = X_Q.shape
        Q, K, V = self._qkv(X_Q, X_K, X_V, offset)
        A = Q @ K.transpose(-2, -1)
        if self.apply_decay:
            D = _get_D(self.gamma, T).unsqueeze(0).to(X_Q.device)
            A *= D
            #A *= self.gamma
        #A = F.softmax(A / math.sqrt(self.inter_size), -1)
        out = A @ V
        out = self._out(X_Q, out)
        return out

    def forward_chunkwise(self, X_Q, X_K=None, X_V=None, S_n=None, att_mask=None, offset=0):
        B, T, C = X_Q.shape
        Q, K, V = self._qkv(X_Q, X_K, X_V, offset)
        D = _get_D(self.gamma, T).to(X_Q.device)
        R_i = (K.transpose(-2, -1) @ (V * D[-1].view(1, T, 1))) + (S_n * (self.gamma ** T))
        
        inner_chunk = Q @ K.transpose(-2, -1)
        if self.apply_decay:
            inner_chunk *= D.unsqueeze(0)
        inner_chunk @= V

        e = torch.zeros(B, T, 1).to(X_Q.device)
        for i in range(T):
            e[..., i, :] = self.gamma ** (i + 1)
        cross_chunk = (Q @ S_n) * e

        out = inner_chunk + cross_chunk
        out = self._out(X_Q, out)
        return out, R_i

    def forward(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        if self.config.forward_method == "parallel":
            y = self.forward_parallel(X_Q, X_KV, X_KV, att_mask, offset)
            return y, S_n
        elif self.config.forward_method == "chunkwise":
            if not self.config.local_recurrence_check:
                return self.forward_chunkwise(X_Q, X_K, X_V, S_n, att_mask, offset)
            T = X_Q.shape[1]
            assert (T % self.config.num_local_chunks) == 0, f"{T} % {self.config.num_local_chunks} != 0"
            nc_step = T // self.config.num_local_chunks
            ret = []
            for i in range(0, T, nc_step):
                os = i + offset
                u = j + nc_step
                am = att_mask[:, j:u] if att_mask is not None else None
                qi = X_Q[:, j:u, :] if X_Q is not None else None
                ki = X_K[:, j:u, :] if X_K is not None else None
                vi = X_V[:, j:u, :] if X_V is not None else None
                y, S_n = self.forward_chunkwise(qi, ki, vi, S_n, am, os)
                ret.append(y)
            y = torch.cat(ret, 1)
            y_par = self.forward_parallel(X_Q, X_KV, X_KV, att_mask, offset)
            print("local check:", torch.allclose(y, y_par, atol=1e-5))
            return y, S_N
        else:
            raise ValueError(f"unknown forward method '{self.config.forward_method}'")


class HStateQKV(nn.Module):
    def __init__(self, config, gamma, in_size, inter_size, out_size, apply_decay_mask=None, apply_attention_mask=False):
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.apply_attention_mask = apply_attention_mask
        self.apply_decay = apply_decay_mask if apply_decay_mask is not None else config.apply_decay
        self.in_size = in_size
        self.inter_size = inter_size
        self.out_size = out_size

        self.proj_Q = nn.Linear(self.in_size, self.inter_size)
        self.proj_K = nn.Linear(self.in_size, self.inter_size)
        self.proj_V = nn.Linear(self.in_size, self.out_size)
        #self.input_gate = nn.Linear(self.in_size, self.inter_size)
        self.input_gate = nn.Linear(self.in_size, config.max_position_embeddings)
        #self.forget_gate = nn.Linear(self.in_size, self.inter_size)
        self.forget_gate = nn.Linear(self.in_size, config.max_position_embeddings)
        self.output_gate = nn.Linear(self.in_size, self.out_size)

        self.rope = RotaryEmbedding(config.rope_dim)
        self.xpos = XPOS(self.inter_size)
        self.act = get_act_func(config.hidden_retention_act)
        self.out_act = get_act_func(config.hidden_out_act)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, X_Q, X_KV=None, S_n=None, att_mask=None, offset=0):
        Q = self.proj_Q(X_Q)
        K = self.proj_K(X_KV)
        V = self.proj_V(X_KV)
        
        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(V)

        Q = self.act(Q)
        K = self.act(K)
        V = self.act(V)

        h, C = S_n

        """
        i = self.input_gate(X_Q)
        i = torch.exp(i)

        f = self.forget_gate(X_Q)
        f = torch.exp(f)

        print(f.shape, C.shape)
        C_t = f * C 
        print(i.shape, (K.transpose(-2, -1) @ V).shape)
        C_t += i * (K.transpose(-2, -1) @ V)
        attn = Q @ C_t

        o = self.output_gate(h)
        o = F.sigmoid(o)

        y = o * attn
        y = self.dropout(y)
        """
        """
        i = self.input_gate(X_Q)
        i = torch.exp(i)

        f = self.forget_gate(X_Q)
        f = torch.exp(f)

        D_tilde = torch.log(f) + i
        m = D_tilde.max(dim=-1).values
        
        #print(D_tilde.shape, m.shape)

        D_ = torch.exp(D_tilde - m.unsqueeze(-1))
        C_t = (Q @ K.transpose(-2, -1)) / (self.inter_size ** .5)
        #print(C_t.shape, D_.shape)
        C_t *= D_

        b = C_t.sum(-1)
        n = torch.maximum(torch.abs(b), torch.exp(-m))
        
        #C_t = C_t * (n**-1).unsqueeze(-1)
        #print(C_t.shape, V.shape)
        y = C_t @ V

        return y, [h, C_t]
        """

        out = Q @ K.transpose(-2, -1) 

        out = out.transpose(-2, -1) @ V

        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)

        return out, S_n


__QKV_CLASS__ = SingleHeadQKV
#__QKV_CLASS__ = MultiHeadQKV
#__QKV_CLASS__ = MultiQueryQKV
#__QKV_CLASS__ = MHeadQKV
#__QKV_CLASS__ = i2D_QKV
#__QKV_CLASS__ = HStateQKV


class EfficientChannelAttention(nn.Module):
    def __init__(self, in_size, N, gamma=2, b=1):
        super().__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool2d(in_size)
        t = int(abs((math.log(N, 2) + self.b) / self.gamma))
        k = t if (t % 2 == 0) else (t + 1)
        self.conv = nn.Conv1d(in_size, in_size, k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        #B, T, N, C = X.shape
        y = self.avg_pool(X)
        #y = y.squeeze(-1)
        y = y.transpose(-2, -1)
        y = self.conv(y)
        y = y.transpose(-2, -1)#.unsqueeze(-1)
        y = self.sigmoid(y)
        y = X * y.expand_as(X)
        return y


class SkipAt(nn.Module):
    def __init__(self, in_size, T):
        super().__init__()
        self.fc = nn.Linear(in_size, in_size*2)
        #self.conv = nn.Conv2d(in_size*2, in_size*2, 1)
        cc = int(T**.5)
        self.conv = nn.Conv2d(cc, cc, 1)
        self.eca_fc = nn.Linear(in_size*2, in_size)
        self.eca = EfficientChannelAttention(T, T)

    def forward(self, X):
        B, T, C = X.shape
        n2d = self.fc(X)
        n2d = n2d.reshape(B, -1, int(T**.5), C*2)
        dwc = self.conv(n2d)
        n2d = dwc.flatten(1, 2)
        #print(dwc.shape, n2d.shape)
        nd = self.eca_fc(n2d)
        nd = self.eca(nd)
        return nd


class AANFFN(nn.Module):
    def __init__(self, io_size, layer_norm_eps):
        super().__init__()
        self.io_size = io_size
        self.inter_size = 3072
        self.core = nn.Sequential(
            nn.Linear(self.io_size, self.inter_size),
            nn.GELU(),
            nn.Linear(self.inter_size, self.io_size)
        )
        self.norm = nn.LayerNorm(io_size, eps=layer_norm_eps)

    def forward(self, X):
        y = self.core(X)
        y = self.norm(y + X)
        return y


class COINBlock(nn.Module):
    def __init__(self, config, gamma, in_size, inter_size, out_size, is_decoder=False, is_cross_encoder=False):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder
        self.is_cross_encoder = is_cross_encoder
        if self.is_decoder:
            assert len(gamma) == 2, gamma
            gamma, s_gamma = gamma
            self.cross_qkv = __QKV_CLASS__(config, s_gamma, out_size, inter_size, out_size)
            self.cross_norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
        self.qkv = __QKV_CLASS__(config, gamma, in_size, inter_size, out_size, apply_attention_mask=is_decoder)#, apply_decay_mask=is_decoder)
        self.norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
        #self.norm = RMSNorm(out_size, eps=config.rms_norm_eps)
        self.hidden_pos_offset = 0

        #self.ffn = AANFFN(out_size, config.layer_norm_eps)

    def forward_encode(self, encoder_query, encoder_hidden_state, residual_query, S_n, att_mask, offset):
        B, T, C = encoder_query.shape
        if S_n is None or self.config.reset_S_n_state:
            S_n = torch.zeros(B, C, C).to(encoder_query.device)
            #S_n = [
            #    torch.zeros(B, C).to(encoder_query.device),
            #    torch.zeros(B, C, C).to(encoder_query.device)
            #]
            self.hidden_pos_offset = 0
        y, s_o = self.qkv(
            X_Q=encoder_query,
            #X_KV=residual_query,
            #X_Q=encoder_hidden_state,
            #X_KV=encoder_hidden_state if self.is_cross_encoder else None,
            S_n=S_n,
            att_mask=att_mask,
            offset=offset
        )
        y = self.norm(y + encoder_query)

        #y = self.ffn(y)

        return y, s_o

    def forward_decode(self, query, cross_query, encoder_hidden_state, residual_query, S_n, att_mask, offset):
        B, T, C = query.shape
        if S_n is None or self.config.reset_S_n_state:
            S_n = [
                torch.zeros(B, C, C).to(query.device),
                torch.zeros(B, C, C).to(query.device)
            ]
            self.hidden_pos_offset = 0
        assert len(S_n) == 2, len(S_n)
        s_1, s_2 = S_n
        y, s_o1 = self.qkv(
            X_Q=query ,#+ residual_query,
            #X_KV=encoder_query,
            #X_KV=residual_query,
            S_n=s_1,
            att_mask=att_mask,
            offset=offset
        )
        y = self.norm(y + query)

        r, s_o2 = self.cross_qkv(
            X_Q=y,
            X_KV=cross_query,
            #X_KV=residual_query,
            S_n=s_2,
            att_mask=att_mask,
            offset=offset
        )
        r = self.cross_norm(r + y)
        s_o = [s_o1, s_o2]
        return r, s_o

    def forward(self, encoder_query, decoder_query, encoder_hidden_state, residual_query, S_n, att_mask, offset):
        B, T, C = encoder_query.shape
        if not self.is_decoder:
            encoder_query, s_o = self.forward_encode(encoder_query, encoder_hidden_state, residual_query, S_n, att_mask, offset)
        else:
            if self.config.revert_decoder:
                decoder_query, s_o = self.forward_decode(decoder_query, encoder_query, encoder_hidden_state, residual_query, S_n, att_mask, offset)
            else:
                encoder_query, s_o = self.forward_decode(encoder_query, decoder_query, encoder_hidden_state, residual_query, S_n, att_mask, offset)
        if self.config.forward_method == "chunkwise" and self.config.apply_hidden_pos_offset:
            self.hidden_pos_offset += T
        return encoder_query, decoder_query, encoder_hidden_state, residual_query, s_o


class COINLayer(nn.Module):
    def __init__(self, i, config, gamma, in_size, inter_size, out_size, num_experts, is_decoder=False, is_cross_encoder=False):
        super().__init__()
        self.num_experts = num_experts
        self.blocks = nn.ModuleList([
            COINBlock(config, gamma, in_size, inter_size, out_size, is_decoder, is_cross_encoder) for _ in range(self.num_experts)
        ])
        print(f"layer {i} num experts: {self.num_experts}")

    def forward(self, encoder_query, decoder_query, encoder_hidden_state, residual_query, S_n, att_mask, offset):
        eq_out, dq_out, ehs_out, rq_out, s_out = [], [], [], [], []
        for B, s_b in zip(self.blocks, S_n):
            eq, dq, ehs, rq, s_i = B(encoder_query, decoder_query, encoder_hidden_state, residual_query, s_b, att_mask, offset)
            eq_out.append(eq)
            dq_out.append(dq)
            ehs_out.append(ehs)
            rq_out.append(rq)
            s_out.append(s_i)
        eq = torch.stack(eq_out).sum(0) #/ self.num_experts
        dq = torch.stack(dq_out).sum(0) #/ self.num_experts
        ehc = torch.stack(ehs_out).sum(0) #/ self.num_experts
        rq = torch.stack(rq_out).sum(0)
        return eq, dq, ehc, rq, s_out


class COINStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_output = config.decoder_output
        self.decoder_schema = config.decoder_schema
        self.num_layers = len(self.decoder_schema)
        self.cross_encoder_schema = config.cross_encoder_schema
        assert len(self.decoder_schema) == len(self.cross_encoder_schema), f"{len(self.decoder_schema)} != {len(self.cross_encoder_schema)}"
        C = config.hidden_size
        self.block_io_schema = config.block_io_schema if config.block_io_schema is not None else [[C, C, C] for _ in range(self.num_layers)]
        self.experts_schema = config.experts_schema if config.experts_schema is not None else [1 for _ in range(self.num_layers)]
        self.num_experts = len(self.experts_schema)
        self.gammas = _create_decay_mask(config, self.decoder_schema)
        """self.layers = nn.ModuleList([
            COINLayer(config, g, ins, ints, outs, e, id, ice) 
            for g, (ins, ints, outs), e, id, ice in zip(
                self.gammas, self.block_io_schema, self.experts_schema, self.decoder_schema, self.cross_encoder_schema
            )
        ])"""
        print(f"num layers: {self.num_layers}")
        print(f"gamma schema: {self.gammas}")
        self.layers = nn.ModuleList([
            COINLayer(i, config, self.gammas[i], *self.block_io_schema[i], self.experts_schema[i], self.decoder_schema[i], self.cross_encoder_schema[i]) for i in range(self.num_layers)
        ])

    def forward(self, encoder_query, decoder_query, encoder_hidden_state, residual_query, S, att_mask, offset):
        if encoder_hidden_state is None:
            encoder_hidden_state = torch.randn(encoder_query.shape).to(encoder_query.device)
        if S is None:
            S = [[None for _ in range(n)] for n in self.experts_schema]
        s_out = []

        eq_out = []
        dq_out = []
        for _ in range(1):
            for L, s_n in zip(self.layers, S):
                encoder_query, decoder_query, encoder_hidden_state, residual_query, s_j = L(encoder_query, decoder_query, encoder_hidden_state, residual_query, s_n, att_mask, offset)
                s_out.append(s_j)
        #        eq_out.append(encoder_query)
        #        dq_out.append(decoder_query)

        #encoder_query = torch.stack(eq_out).sum(0) / 4
        #decoder_query = torch.stack(dq_out).sum(0) / 4

        if self.decoder_output == "strict":
            out = decoder_query
        elif self.decoder_output == "adaptive":
            out = encoder_query + decoder_query
        elif self.decoder_output == "none" or self.decoder_output is None:
            out = encoder_query
        else:
            raise ValueError()
        encoder_hidden_state = encoder_query
        return out, encoder_hidden_state, s_out


class COINPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

    def forward(self, X):
        X = X[..., 0, :]
        X = self.dense(X)
        X = self.act(X)
        return X


class COINModel(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder_embeddings = COINEmbeddings(config)
        self.decoder_embeddings = COINEmbeddings(config)
        self.residual_embeddings = COINEmbeddings(config)
        self.stack = COINStack(config)
        self.pooler = COINPooler(config)
        self.post_init()
    
    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, **kwargs):
        if self.config.disable_teacher_forcing or not self.training:
            if self.config.switch_ii_decoder_ii:
                input_ids = decoder_input_ids
                inputs_embeds = decoder_inputs_embeds
            else:
                decoder_input_ids = input_ids
                decoder_inputs_embeds = inputs_embeds
        encoder_query = self.encoder_embeddings(input_ids, inputs_embeds)
        decoder_query = self.decoder_embeddings(decoder_input_ids, decoder_inputs_embeds)
        residual_query = self.residual_embeddings(decoder_input_ids, decoder_inputs_embeds)
        #residual_query = self.residual_embeddings(input_ids, inputs_embeds)
        out, encoder_hidden_state, S = self.stack(encoder_query, decoder_query, encoder_hidden_state, residual_query, S, attention_mask, offset)
        pooled_out = self.pooler(out)
        return out, pooled_out, S, encoder_hidden_state


class COINForSequenceClassification(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss() if self.config.num_labels <= 2 else nn.BCEWithLogitsLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        _, pooled_out, S, encoder_hidden_state = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)
        logits = self.dropout(pooled_out)
        logits = self.classifier(logits)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C,
            loss
        )


class COINForHierachicalClassification(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        _, pooled_out, S, encoder_hidden_state = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)
        logits = self.dropout(pooled_out)
        logits = self.classifier(logits)
        #loss = self.loss_fn(logits, labels) if labels is not None else None
        l1_log, l2_log = logits.chunk(2, -1)
        l1_lab, l2_lab = labels.chunk(2, -1)
        l1_loss = self.loss_fn(l1_log, l1_lab) 
        l2_loss = self.loss_fn(l2_log, l2_lab) 
        loss = l1_loss + l2_loss
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C,
            loss
        )


class COINPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_act_func(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, X):
        X = self.dense(X)
        X = self.transform_act_fn(X)
        X = self.layer_norm(X)
        return X


class COINLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = COINPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        # former self.decoder.biad

    def forward(self, X):
        X = self.transform(X)
        X = self.decoder(X)
        return X


class COINForMaskedLM(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config)
        self.cls = COINLMPredictionHead(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        out, _, S, encoder_hidden_state = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)

        logits = self.cls(out)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)) if labels is not None else None
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C,
            loss
        )


class COINForConditionalGeneration(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        #decoder_input_ids = _shift_right(decoder_input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        out, _, S, encoder_hidden_state = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)
        logits = self.lm_head(out)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)) #if labels is not None else None
        #if aux_loss is not None:
        #    loss += aux_loss
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C,
            loss
        )


class COINForBucketSort(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #self.coin = COINModel(config)
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.stack = COINStack(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        #decoder_input_ids = _shift_right(decoder_input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        #logits, _, S, encoder_hidden_state = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)
        #emb = F.one_hot(input_ids, num_classes=self.config.vocab_size).float()
        #dec_emb = F.one_hot(decoder_input_ids, num_classes=self.config.vocab_size).float()
        emb = self.embeddings(F.one_hot(input_ids, self.config.vocab_size).float())
        dec_emb = self.decoder_embeddings(F.one_hot(decoder_input_ids, self.config.vocab_size).float())
        logits, encoder_hidden_state, S = self.stack(emb, dec_emb, encoder_hidden_state, dec_emb, S, attention_mask, offset)
        logits = self.lm_head(logits)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        #loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.argmax(-1).view(-1))
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C, 
            loss
        )
    

class COINForParityCheck(COINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.stack = COINStack(config)
        self.pooler = COINPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss() if self.config.num_labels <= 2 else nn.BCEWithLogitsLoss()
        self.post_init()
    
    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        emb = self.embeddings(F.one_hot(input_ids, self.config.vocab_size).float())
        dec_emb = self.decoder_embeddings(F.one_hot(decoder_input_ids, self.config.vocab_size).float())
        logits, encoder_hidden_state, S = self.stack(emb, dec_emb, encoder_hidden_state, dec_emb, S, attention_mask, offset)
        pooled_out = self.pooler(logits)
        logits = self.dropout(pooled_out)
        logits = self.classifier(logits)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C,
            loss
        )
