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


def print_tensor_list_shape(X, prefix=""):
    if isinstance(X, torch.Tensor):
        print(f"{prefix}[\n    {X.shape}")
    if isinstance(X, list):
        for n in X:
            print_tensor_list_shape(n, f"{prefix    }")
            print(f"{prefix}]")

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
            revert_decoder: bool = False,
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


class QKV(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, apply_decay_mask=None):
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
        self.proj_biad = nn.Parameter(torch.randn(self.out_size) / self.out_size)
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

    def forward(self, X_Q, X_K=None, X_V=None, S_n=None, att_mask=None, offset=0):
        if self.config.forward_method == "parallel":
            y = self.forward_parallel(X_Q, X_K, X_V, att_mask, offset)
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
            y_par = self.forward_parallel(X_Q, X_K, X_V, att_mask, offset)
            print("local check:", torch.allclose(y, y_par, atol=1e-5))
            return y, S_N
        else:
            raise ValueError(f"unknown forward method '{self.config.forward_method}'")


class EncoderBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, is_cross_encoder=False):
        super().__init__()
        self.config = config
        self.is_cross_encoder = is_cross_encoder
        self.qkv = QKV(config, gamma, in_size, out_size)
        self.norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
        self.hidden_pos_offset = 0

    def forward(self, X, S_n=None, att_mask=None, offset=0):
        X_e, X_h = X[:2]
        B, T, C = X_e.shape
        if S_n is None or self.config.reset_S_n_state:
            S_n = torch.zeros(B, C, C).to(X_e.device)
            self.hidden_pos_offset = 0

        y, S_n = self.qkv(
            X_Q=X_e,
        #    X_K=X_h if self.is_cross_encoder else None,
        #    X_V=X_h if self.is_cross_encoder else None,
            S_n=S_n,
            att_mask=att_mask,
            offset=offset + self.hidden_pos_offset
        )
        y = self.norm(y + X_e)
        o = X.clone()
        o[0] = y
        if self.config.apply_hidden_pos_offset and self.config.forward_method == "chunkwise":
            self.hidden_pos_offset += T
        return o, S_n


class DecoderBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size):
        super().__init__()
        self.config = config
        assert len(gamma) == 2, gamma
        self.qkv = QKV(config, gamma[0], in_size, in_size)
        self.cross_qkv = QKV(config, gamma[1], in_size, out_size)
        self.norm = nn.LayerNorm(in_size, eps=config.layer_norm_eps)
        self.cross_norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
        self.hidden_pos_offset = 0

    def forward(self, X, S_n=None, att_mask=None, offset=0):
        X_e, X_d, X_r = X[0], X[2], X[3]
        B, T, C = X_d.shape
        if S_n is None or self.config.reset_S_n_state:
            S_n = [
                torch.zeros(B, C, C).to(X_d.device),
                torch.zeros(B, C, C).to(X_d.device)
            ]
            self.hidden_pos_offset = 0
        assert len(S_n) == 2, f"{S_n.__class__} {len(S_n)}"
        s_1, s_2 = S_n
        y, s_1 = self.qkv(
            X_Q=X_d,
            S_n=s_1,
            att_mask=att_mask,
            offset=offset + self.hidden_pos_offset
        )
        y = self.norm(y + X_d)

        z, s_2 = self.cross_qkv(
            X_Q=y,
            X_K=X_e,
            X_V=X_e,
            S_n=s_2,
            att_mask=att_mask,
            offset=offset + self.hidden_pos_offset
        )
        z = self.cross_norm(z + y)
        o = X.clone()
        o[2] = z

        if self.config.apply_hidden_pos_offset and self.config.forward_method == "chunkwise":
            self.hidden_pos_offset += T
        return o, [s_1, s_2]


class AANFFN(nn.Module):
    def __init__(self, config, in_size, inter_size, out_size):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(in_size, inter_size),
            get_act_func(config.hidden_act),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(inter_size, out_size)
        )
        self.norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
    
    def forward(self, X):
        y = self.core(X)
        y = self.norm(y + X)
        return y


class COINBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, is_decoder=False, is_cross_encoder=False):
        super().__init__()
        self.in_size = in_size if in_size is not None else config.hidden_size
        self.out_size = out_size if out_size is not None else config.hidden_size
        self.config = config
        self.gamma = gamma
        self.is_decoder = is_decoder
        self.is_cross_encoder = is_cross_encoder
        if self.is_decoder:
            self.block = DecoderBlock(self.config, self.gamma, self.in_size, self.out_size)
        else:
            self.block = EncoderBlock(self.config, self.gamma, self.in_size, self.out_size, self.is_cross_encoder)

    def forward(self, X, S_n=None, att_mask=None, offset=0):
        y, S_n = self.block(X, S_n, att_mask, offset)
        return y, S_n


#class COINRegionHolder(COINPreTrainedModel):
class COINRegionHolder(nn.Module):
    def __init__(self, config):
        #super().__init__(config)
        super().__init__()
        self.config = config
        self.decoder_schema = config.decoder_schema
        self.cross_encoder_schema = config.cross_encoder_schema
        self.gammas = _create_decay_mask(config, schema=self.decoder_schema)
        print("decay mask schema:", self.gammas)
        self.io_schema = config.block_io_schema if config.block_io_schema is not None else [None for _ in range(len(self.gammas) + 1)]
        assert len(self.decoder_schema) == len(self.gammas) == len(self.cross_encoder_schema) == (len(self.io_schema) - 1), \
            f"{len(self.decoder_schema)} != {len(self.gammas)} != {len(self.cross_encoder_schema)} != ({len(self.io_schema)} - 1)" 
        self.blocks = nn.ModuleList([
            COINBlock(self.config, self.gammas[i], self.io_schema[i], self.io_schema[i+1], self.decoder_schema[i], self.cross_encoder_schema[i])
            for i in range(len(self.gammas))
        ])
    
    def __getitem__(self, i):
        return self.blocks[i]

    def __setitem__(self, k, v):
        self.blocks[k] = v

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return self.blocks.__iter__()


class COINSamplerLayer(nn.ModuleList):
    def __init__(self, config, blocks):
        super().__init__()
        if isinstance(blocks, nn.ModuleList):
            self.blocks = blocks
        elif isinstance(blocks, list) or isinstance(blocks, tuple):
            self.blocks = nn.ModuleList(blocks)
        else:
            raise ValueError(blocks.__class__)
        self.num_blocks = len(self.blocks)

    def forward(self, X, S_n=None, att_mask=None, offset=0):
        """
        X[0]: encoder query
        X[1]: encoder hidden state
        X[2]: decoder query
        X[3]: residual query
        """
        assert X.shape[1] == 4, X.shape
        assert X.shape[0] == self.num_blocks, f"{X.shape}[1] != {self.num_blocks}"

        if S_n is None:
            S_n = [None for _ in range(self.num_blocks)]
        
        y_out, S_out = [], []
        for B, X_i, S_i in zip(self.blocks, X, S_n):
            y_i, s_j = B(X_i, S_i, att_mask, offset)
            y_out.append(y_i)
            S_out.append(s_j)
        
        y = torch.stack(y_out, 0)
        aux_loss = None
        return y, S_out, aux_loss


class COINSampler(nn.ModuleList):
    def __init__(self, config, regions):
        super().__init__()
        self.config = config
        self.share_S = config.share_S
        self.decoder_output = config.decoder_output
        self.layers = nn.ModuleList([
            COINSamplerLayer(config, n) for n in zip(*regions)
        ])
        self.num_layers = len(self.layers)
        self.num_regions = len(regions)

    def forward(self, encoder_query, decoder_query, residual_query, encoder_hidden_state, S_n=None, att_mask=None, offset=0):
        B, T, C = encoder_query.shape
        encoder_query = encoder_query.unsqueeze(0).repeat(self.num_regions, 1, 1, 1)
        decoder_query = decoder_query.unsqueeze(0).repeat(self.num_regions, 1, 1, 1)
        residual_query = residual_query.unsqueeze(0).repeat(self.num_regions, 1, 1, 1)
        if encoder_hidden_state is None:
            #encoder_hidden_state = torch.zeros(self.num_regions, B, T, C).to(encoder_query.device)
            #encoder_hidden_state = [None for _ in range(self.num_regions)]
            encoder_hidden_state = torch.randn(self.num_regions, B, T, C).to(encoder_query.device) #for _ in range(self.num_regions)]
        X = torch.stack((encoder_query, encoder_hidden_state, decoder_query, residual_query), 1)
        if self.share_S:
            s_n = S_n
        else:
            if S_n is None:
                S_n = [None for _ in range(self.num_layers)]
            S_out = []
        aux_loss = None
        for i, L in enumerate(self.layers):
            if not self.share_S:
                s_n = S_n[i]
            X, s_n, a_l = L(X, s_n, att_mask, offset)
            if not self.share_S:
                S_out.append(s_n)
            if aux_loss is None:
                aux_loss = a_l
            else:
                aux_loss += a_l
        S_n = s_n if self.share_S else S_out
        if self.decoder_output == "strict":
            out = X[:, 2]
        elif self.decoder_output == "adaptive":
            out = X[:, 0] + X[:, 2]
        elif self.decoder_output == "none":
            out = X[:, 0]
        out = out.sum(0)
        return out, X[:, 1], S_n, aux_loss


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
    def __init__(self, config, region_config=None):
        super().__init__(config)
        self.config = config
        self.disable_teacher_forcing = config.disable_teacher_forcing
        self.num_regions = config.num_regions
        self.regions = nn.ModuleList([
            COINRegionHolder(config) for _ in range(self.num_regions)
        ])
        self.sampler = COINSampler(config, self.regions)
        self.encoder_embeddings = COINEmbeddings(config)
        self.decoder_embeddings = COINEmbeddings(config)
        self.residual_embeddings = COINEmbeddings(config)
        self.pooler = COINPooler(config)
        print("num core parameters: {:,}".format(num_parameters(self)))
        self.post_init()

    def _sample(self, input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0):
        encoder_query = self.encoder_embeddings(input_ids, inputs_embeds)
        if self.disable_teacher_forcing or not self.training:
            decoder_input_ids = input_ids
            decoder_inputs_embeds = inputs_embeds
        decoder_query = self.decoder_embeddings(decoder_input_ids, decoder_inputs_embeds)
        residual_query = self.residual_embeddings(decoder_input_ids, decoder_inputs_embeds)
        return self.sampler(encoder_query, decoder_query, residual_query, encoder_hidden_state, S, attention_mask, offset)

    def _chunk_sample(self, input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0):
        T = input_ids.shape[1]
        assert (T % self.config.num_global_chunks) == 0, f"{T} % {self.config.num_global_chunks} != 0"
        nc_step = T // self.config.num_global_chunks
        Y = []
        aux_loss = None
        for j in range(0, T, nc_step):
            i_ = j + offset
            u = j + nc_step
            ii = input_ids[:, j:u] if input_ids is not None else None
            dii = decoder_input_ids[:, j:u] if decoder_input_ids is not None else None
            ie = inputs_embeds[:, j:u, :] if inputs_embeds is not None else None
            die = decoder_inputs_embeds[:, j:u, :] if decoder_inputs_embeds is not None else None
            am = attention_mask[:, j:u] if attention_mask is not None else None

            y, encoder_hidden_state, S, a_u = self._sample(ii, dii, encoder_hidden_state, S, ie, die, am, i_)
            Y.append(y)
            if aux_loss is None:
                aux_loss = a_u
            else:
                aux_loss += a_u
        Y = torch.cat(Y, 1)
        return Y, encoder_hidden_state, S, aux_loss

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, **kwargs):
        if not self.config.global_recurrence_check:
            out, encoder_hidden_state, S, aux_loss = self._sample(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
        else:
            self.config.forward_method = "parallel"
            y_par, ehs_par, S_par, al_par = self._sample(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
            self.config.forward_method = "chunkwise"
            y_chn, ehs_chn, S_chn, al_chn = self._chunk_sample(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
            chrs = torch.allclose(y_par, y_chn, atol=1e-5)
            print("global check:", chrs, "SUCCESS" if chrs else "FAILED")
            out, encoder_hidden_state, S, aux_loss = y_chn, ehs_chn, S_chn, al_chn
        pooled_out = self.pooler(out)
        return out, pooled_out, S, encoder_hidden_state, aux_loss


class COINForSequenceClassification(COINPreTrainedModel):
    def __init__(self, config, region_config=None):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config, region_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss() if self.config.num_labels <= 2 else nn.BCEWithLogitsLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        _, pooled_out, S, encoder_hidden_state, aux_loss = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)
        logits = self.dropout(pooled_out)
        logits = self.classifier(logits)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        if aux_loss is not None:
            loss += aux_loss
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
    def __init__(self, config, region_config=None):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config, region_config)
        self.cls = COINLMPredictionHead(config)
        #self.cls_heads = nn.ModuleList([
        #    COINLMPredictionHead(config) for _ in range(config.num_repetitions)
        #])
        #self.cls = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()#reduction="none")
        #self.label_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        #decoder_input_ids = _shift_right(decoder_input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        out, _, S, encoder_hidden_state, aux_loss = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)

        logits = self.cls(out)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)) if labels is not None else None
        if aux_loss is not None:
            loss += aux_loss
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C,
            loss
        )


class COINForConditionalGeneration(COINPreTrainedModel):
    def __init__(self, config, region_config=None):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config, region_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, C, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        #decoder_input_ids = _shift_right(decoder_input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        out, _, S, encoder_hidden_state, aux_loss = self.coin(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset, **kwargs)
        logits = self.lm_head(out)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)) if labels is not None else None
        if aux_loss is not None:
            loss += aux_loss
        return COINOutputClass(
            logits,
            encoder_hidden_state,
            S,
            C,
            loss
        )
