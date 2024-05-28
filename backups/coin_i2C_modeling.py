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
            share_S: bool = True,  # share S between layers
            decoder_output: str = "adaptive",  # adaptive, strict or none
            decoder_schema: List[bool] = [0, 1],
            cross_encoder_schema: List[bool] = [1, 0],
            apply_softmax_gate: bool = True,
            revert_decoder: bool = False,
            disable_teacher_forcing: bool = False,
            allow_encoder_teacher_forcing: bool = False,
            block_ioh_schema: Optional[List[int]] = None,
            num_repetitions: int = 1,
            apply_hidden_pos_offset: bool = False,
            chunkwise_num_chunks: int = 1,
            apply_chunking_globally: bool = True,
            print_checks: bool = False,
            xdec_main_switch: bool = False,
            R_skip_connection: bool = False,
            multi_head_qkv: bool = False,
            fixed_decay_value: Optional[float] = None,
            reset_S_n_state: bool = False,
            rms_norm_eps: float = 1e-8,
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
        #        self.apply_block_wise_ffn = apply_block_wise_ffn
        #        self.apply_layer_wise_ffn = apply_layer_wise_ffn
        #        self.apply_hierarchical_ffn = apply_hierarchical_ffn
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
        self.block_ioh_schema = block_ioh_schema
        self.num_repetitions = num_repetitions
        self.apply_hidden_pos_offset = apply_hidden_pos_offset
        self.chunkwise_num_chunks = chunkwise_num_chunks
        self.apply_chunking_globally = apply_chunking_globally
        self.print_checks = print_checks
        self.xdec_main_switch = xdec_main_switch
        self.R_skip_connection = R_skip_connection
        self.multi_head_qkv = multi_head_qkv
        self.fixed_decay_value = fixed_decay_value
        self.reset_S_n_state = reset_S_n_state
        self.rms_norm_eps = rms_norm_eps


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

        self.filter_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(self, input_ids, inputs_embeds=None):
        E = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds

        #emb_f = self.filter_embeddings(input_ids)
        #E *= F.softmax(emb_f, dim=1)

        X = self.norm(E)
        X = self.dropout(X)
        return X


class WQKV(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, fixed_WQ=False, mask=False, apply_decay=None):
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.mask = mask
        self.apply_decay = config.apply_decay if apply_decay is None else apply_decay
        self.multi_head = config.multi_head_qkv
        self.num_heads = config.num_heads
        self.in_size = in_size
        self.out_size = out_size
        assert (config.hidden_size % self.num_heads) == 0, f"({config.hidden_size} % {self.num_heads}) != 0"
        if self.multi_head:
            self.head_size = config.hidden_size // self.num_heads
        else:
            self.head_size = config.hidden_size

        self.fixed_WQ = fixed_WQ
        if self.fixed_WQ:
            self.mpe = config.max_position_embeddings
            self.W_Fq = nn.Parameter(torch.randn(self.mpe, self.in_size) / self.mpe)
        else:
            self.W_Q = nn.Parameter(torch.randn(self.in_size, self.in_size) / self.in_size)
        self.W_K = nn.Parameter(torch.randn(self.in_size, self.in_size) / self.in_size)
        self.W_V = nn.Parameter(torch.randn(self.in_size, self.out_size) / self.in_size)

        #self.xpos = XPOS(self.in_size)
        self.rope = RotaryEmbedding(self.num_heads)
        self.act = get_act_func(config.hidden_retention_act)
        self.out_act = get_act_func(config.hidden_out_act)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.layer_norm = nn.LayerNorm(self.head_size, eps=config.layer_norm_eps)
        self.rms_norm = RMSNorm(self.head_size, eps=config.rms_norm_eps)                                   
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.W_O = nn.Parameter(torch.randn(self.out_size, self.out_size))

    def _to_scores(self, X):
        B, T = X.shape[:2]
        return X.view(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    def _qkv(self, X, KV_in, offset):
        #X = X.transpose(-2, -1).reshape(X.shape)

        if KV_in is not None:
            K = KV_in @ self.W_K
            V = KV_in @ self.W_V
        else:
            K = X @ self.W_K
            V = X @ self.W_V
        if self.fixed_WQ:
            Q = self.W_Fq
        else:
            Q = X @ self.W_Q

        if self.multi_head:
            raise NotImplementedError("WHAT THE FUCK IS MULTI HEAD")
            Q = self._to_scores(Q)
            K = self._to_scores(K)
            V = self._to_scores(V)

        #Q = Q.unsqueeze(2)
        #K = K.unsqueeze(2)
        #V = V.unsqueeze(2)

        Q = self.rope.rotate_queries_or_keys(Q, offset=offset)
        K = self.rope.rotate_queries_or_keys(K, offset=offset)
        #Q = self.xpos(Q)
        #K = self.xpos(K, downscale=True)

        if self.act is not None:
            Q = self.act(Q)
            K = self.act(K)
            V = self.act(V)
        return Q, K, V

    def _out(self, out):
        B, T = out.shape[0], out.shape[-2]
        #out = out.transpose(-2, -1).contiguous().view(B, T, self.out_size)
  
        out = self.dropout(out)
        #out = self.rms_norm(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = self.dropout(out)
        
        out = out.transpose(-2, -1).contiguous().view(B, T, self.out_size)
        
        return out

    def forward_parallel(self, Q_in, KV_in, att_mask, offset):
        B, T, C = Q_in.shape
        Q, K, V = self._qkv(Q_in, KV_in, offset)

        QK = Q @ K.transpose(-2, -1)
        #QK = QK.transpose(-2, -1)
        if self.apply_decay:
            D = _get_D(self.gamma, T).unsqueeze(0).to(Q_in.device)
            QK *= D
            #QK = QK * F.sigmoid(self.W_decay)
            #QK *= self.gamma

        #QK = QK.transpose(-2, -1)
        
        #if self.mask:
        #    QK *= att_mask.view(B, 1, T).repeat(1, T, 1)
        
        out = QK @ V
        out = self._out(out)
        return out

    def leg_forward_chunkwise(self, Q_in, KV_in, S_n, att_mask, offset):
        B, T, C = Q_in.shape
        Q, K, V = self._qkv(Q_in, KV_in, offset)
        D = _get_D(self.gamma, T).to(Q_in.device)

        R_i = K.transpose(-2, -1) @ (V * D[-1].view(1, T, 1))
        R_i += (self.gamma ** T) * S_n

        inner_chunk = Q @ K.transpose(-2, -1)
        if self.apply_decay:
            inner_chunk *= D.unsqueeze(0)
        #if self.mask:
        #    inner_chunk *= att_mask.view(B, 1, T).repeat(1, T, 1)
        inner_chunk @= V

        e = torch.zeros(B, T, 1).to(Q_in.device)
        for _i  in range(T):
            e[:, _i, :] = self.gamma ** (_i + 1)
        cross_chunk = (Q @ S_n) * e

        out = inner_chunk + cross_chunk
        out = self._out(out)
        return out, R_i
            
    def forward_chunkwise(self, Q_in, KV_in, S_n, att_mask, offset):
        B, T, C = Q_in.shape
        Q, K, V = self._qkv(Q_in, KV_in, offset)
        D = _get_D(self.gamma, T).to(Q_in.device)

        R_i = (K).transpose(-2, -1) @ (V * D[-1].view(1, T, 1))
        R_i += S_n * (self.gamma ** T)

        inner_chunk = Q @ K.transpose(-2, -1)
        if self.apply_decay:
            inner_chunk *= D.unsqueeze(0)
        #if self.mask:
        #    inner_chunk *= att_mask.view(B, 1, T).repeat(1, T, 1)
        inner_chunk @= V

        e_shape = (B, self.num_heads, T, 1) if self.multi_head else (B, T, 1)
        e = torch.zeros(e_shape).to(Q_in.device)
        for _i  in range(T):
            e[..., _i, :] = self.gamma ** (_i + 1)
        cross_chunk = (Q @ S_n) * e

        out = inner_chunk + cross_chunk
        #out = self._out(out)
        return out, R_i

    def forward(self, Q_in, KV_in=None, S_n=None, att_mask=None, offset=0):
        if self.config.forward_method == "parallel":
            y = self.forward_parallel(Q_in, KV_in, att_mask, offset)
            return y, S_n
        elif self.config.forward_method == "chunkwise":
            if self.config.apply_chunking_globally:
                return self.forward_chunkwise(Q_in, KV_in, S_n, att_mask, offset)
            
            T = Q_in.shape[-2]
            assert (T % self.config.chunkwise_num_chunks) == 0, f"{T} % {self.config.chunkwise_num_chunks} != 0"
            #nc_step = T // (self.config.chunkwise_num_chunks * 2)
            nc_step = T // self.config.chunkwise_num_chunks
            ret = []
            for j in range(0, T, nc_step):
                #print("CALL", j, T, nc_step)
                i_ = j + offset
                u = j + nc_step
                am = att_mask[:, j:u] if att_mask is not None else None
                qi = Q_in[:, j:u] if Q_in is not None else None
                kvi = KV_in[:, j:u] if KV_in is not None else None
                y, S_n = self.forward_chunkwise(qi, kvi, S_n, am, i_)
                ret.append(y)
            Y = torch.cat(ret, 1)

            if self.config.print_checks:
                Y_par = self.forward_parallel(Q_in, KV_in, att_mask, offset)
                print("local check:", torch.allclose(Y_par, Y, atol=1e-5))

            return Y, S_n
        else:
            raise ValueError(self.config.forward_method)


class EncoderBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, is_cross_encoder):
        super().__init__()
        self.config = config
        self.is_cross_encoder = is_cross_encoder
        self.wqkv = WQKV(config, gamma, in_size, out_size)#, fixed_WQ=not is_cross_encoder)
        self.out_norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
        #self.out_norm = RMSNorm(out_size, eps=config.rms_norm_eps)
        self.in_norm = RMSNorm(out_size, eps=config.rms_norm_eps)
        self.num_heads = config.num_heads
        self.multi_head = config.multi_head_qkv
        assert (config.hidden_size % self.num_heads) == 0, f"({config.hidden_size} % {self.num_heads}) != 0"
        self.head_size = config.hidden_size // self.num_heads
        self.hidden_pos_offset = 0

    def forward(self, query, hidden_state, S_n, att_mask=None, offset=0):
        B, T, C = query.shape
        if S_n is None or self.config.reset_S_n_state:
            S_shape = (B, self.config.num_heads, self.head_size, self.head_size) if self.multi_head else (B, C, C)
            S_n = torch.zeros(S_shape).to(query.device)
            self.hidden_pos_offset = 0

        #if hidden_state is None:
        #    hidden_state = torch.randn(query.shape).to(query.device)

        #query = self.in_norm(query)
        y, S_n = self.wqkv(
        #y = self.wqkv.forward_parallel(
            Q_in=query,
        #    Q_in=hidden_state if self.is_cross_encoder else query,
        #    KV_in=hidden_state if self.is_cross_encoder else None,
            #Q_in=hidden_state,
            #KV_in=hidden_state,

            S_n=S_n,
            att_mask=att_mask,
            offset=offset + self.hidden_pos_offset
        )
        #y += query
        y = self.out_norm(y + query)
        if self.config.apply_hidden_pos_offset and self.config.forward_method == "chunkwise":
            self.hidden_pos_offset += T
        return y, S_n


class DecoderBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size):
        super().__init__()
        self.config = config
        assert len(gamma) == 2, gamma
        self.wqkv = WQKV(config, gamma[0], in_size, in_size)#, apply_decay=False)#, fixed_WQ=True)
        self.cross_wqkv = WQKV(config, gamma[1], in_size, out_size)
        #self.inner_norm = nn.LayerNorm(in_size, eps=config.layer_norm_eps)
        #self.cross_norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
        self.inner_norm = RMSNorm(in_size, eps=config.rms_norm_eps)
        self.cross_norm = RMSNorm(out_size, eps=config.rms_norm_eps)
        self.hidden_pos_offset = 0
    
    def forward(self, encoder_query, decoder_query, S_ns, att_mask=None, offset=0):
        B, T, C = decoder_query.shape
        if S_ns is None:
            S_ns = [
                torch.zeros(B, C, C).to(encoder_query.device),
                torch.zeros(B, C, C).to(encoder_query.device)
            ]
            self.hidden_pos_offset = 0

        assert len(S_ns) == 2, len(S_ns)
        s_1, s_2 = S_ns
        decoder_query = self.inner_norm(decoder_query)
        y, s_1 = self.wqkv(
            Q_in=decoder_query,
            S_n=s_1,
            att_mask=att_mask,
            offset=offset + self.hidden_pos_offset
        )
        #y = self.inner_norm(y + decoder_query)
        y += decoder_query
        #y = self.cross_norm(y)
        encoder_query = self.cross_norm(encoder_query)
        z, s_2 = self.cross_wqkv(
            Q_in=y,
            KV_in=encoder_query,
            S_n=s_2,
            att_mask=att_mask,
            offset=offset + self.hidden_pos_offset
        )
        #z = self.cross_norm(z + y)
        z += y
        if self.config.apply_hidden_pos_offset and self.config.forward_method == "chunkwise":
            self.hidden_pos_offset += T
        return z, [s_1, s_2]


class COINBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, is_decoder, is_cross_encoder):
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
    
    def forward(self, encoder_query, decoder_query, encoder_hidden_state, S_ns, att_mask=None, offset=0):
        if self.is_decoder:
            decoder_query, S_ns = self.block(encoder_query, decoder_query, S_ns, att_mask, offset)
        else:
            encoder_query, S_ns = self.block(encoder_query, encoder_hidden_state, S_ns, att_mask, offset)
            
        return encoder_query, decoder_query, S_ns



class NBWAttention(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, fixed_WQ=False, mask=False, apply_decay=None):
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.apply_decay = config.apply_decay if apply_decay is None else apply_decay
        self.out_size = out_size
        self.W_Q = nn.Parameter(torch.randn(in_size, in_size) / in_size)
        self.W_K = nn.Parameter(torch.randn(in_size, in_size) / in_size)
        self.W_V = nn.Parameter(torch.randn(in_size, out_size) / in_size)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels, eps=config.retention_group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = get_act_func(config.hidden_retention_act)
        #self.rope = RotaryEmbedding(in_size)
        self.xpos = XPOS(in_size)
    
    def _qkv(self, Q_in, KV_in, offset):
        if KV_in is not None:
            K = KV_in @ self.W_K
            V = KV_in @ self.W_V
        else:
            K = Q_in @ self.W_K
            V = Q_in @ self.W_V
        Q = Q_in @ self.W_Q

        #Q = self.rope(Q, offset=offset)
        #K = self.rope(K, offset=offset)
        #Q = self.xpos(Q, offset=offset)
        #K = self.xpos(K, offset=offset, downscale=True)

        if self.act is not None:
            Q = self.act(Q)
            K = self.act(K)
            V = self.act(V)
        return Q, K, V

    def _out(self, out):
        B, T = out.shape[:2]
        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        out = out.transpose(-2, -1).contiguous().view(B, T, self.out_size)
        return out

    def forward_parallel(self, Q_in, KV_in=None, att_mask=None, offset=0):
        B, T, C = Q_in.shape
        Q, K, V = self._qkv(Q_in, KV_in, offset)

        QK = Q @ K.transpose(-2, -1)
        if self.apply_decay:
            D = _get_D(self.gamma, T).unsqueeze(0).to(Q_in.device)
            QK *= D
        out = QK @ V
        out = self._out(out)
        return out

    def forward(self, Q_in, KV_in=None, S_n=None, att_mask=None, offset=0):
        if self.config.forward_method == "parallel":
            y = self.forward_parallel(Q_in, KV_in, att_mask, offset)
            return y, S_n

        else:
            raise NotImplementedError()

    
class NBWEncoderBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, fixed_WQ=False, mask=False, apply_decay=True):
        super().__init__()
        self.qkv = NBWAttention(config, gamma, in_size, out_size, fixed_WQ, mask, apply_decay)
        self.layer_norm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)

    def forward(self, Q_in, KV_in=None, S_ns=None, att_mask=None, offset=0):
        y, S_ns = self.qkv(
            Q_in=Q_in,
            KV_in=KV_in,
            S_n=S_ns,
            att_mask=att_mask,
            offset=offset
        )
        y = self.layer_norm(y + Q_in)
        return y, S_ns


class COINNBWBlock(nn.Module):
    def __init__(self, config, gamma, in_size, out_size, is_decoder, is_cross_encoder):
        super().__init__()
        self.in_size = in_size if in_size is not None else config.hidden_size
        self.out_size = out_size if out_size is not None else config.hidden_size
        self.config = config
        self.gamma = gamma
        self.is_decoder = is_decoder
        self.is_cross_encoder = is_cross_encoder
        if self.is_decoder:
            self.block = NBWDecoderBlock(self.config, self.gamma, self.in_size, self.out_size)
        else:
            self.block = NBWEncoderBlock(self.config, self.gamma, self.in_size, self.out_size, self.is_cross_encoder)
    
    def forward(self, encoder_query, decoder_query, encoder_hidden_state, S_ns, att_mask=None, offset=0):
        if self.is_decoder:
            decoder_query, S_ns = self.block(encoder_query, decoder_query, S_ns, att_mask, offset)
        else:
            encoder_query, S_ns = self.block(encoder_query, encoder_hidden_state, S_ns, att_mask, offset)
            
        return encoder_query, decoder_query, S_ns


__COINBlock__ = COINBlock
#__COINBlock__ = COINNBWBlock


#class COINRegionHolder(COINRegionPreTrainedModel):
class COINRegionHolder(nn.Module):
    def __init__(self, config):
        #super().__init__(config)
        super().__init__()
        self.config = config
        self.decoder_schema = config.decoder_schema
        self.cross_encoder_schema = config.cross_encoder_schema
        #self.gammas = _create_decay_mask(config, schema=self.decoder_schema)
        self.gammas = _create_encoder_decoder_decay_mask(config, self.decoder_schema)
        print("decay mask schema:", self.gammas)
        self.ioh_schema = config.block_ioh_schema
        assert len(self.decoder_schema) == len(self.gammas) == len(self.cross_encoder_schema) == (len(self.ioh_schema) - 1), \
                f"{len(self.decoder_schema)} != {len(self.gammas)} != {len(self.cross_encoder_schema)} != ({len(self.ioh_schema)} - 1)"
        self.blocks = nn.ModuleList([
            __COINBlock__(config, self.gammas[i], self.ioh_schema[i], self.ioh_schema[i+1], self.decoder_schema[i], self.cross_encoder_schema[i]) 
            for i in range(len(self.decoder_schema))
        ])
        #self.post_init()

    def __getitem__(self, i):
        return self.blocks[i]

    def __setitem__(self, k, v):
        self.blocks[k] = v

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return self.blocks.__iter__()


class COINSamplerLayer(nn.Module):
    def __init__(self, config, blocks):
        super().__init__()
        if isinstance(blocks, nn.ModuleList):
            self.blocks = blocks
        elif isinstance(blocks, list) or isinstance(blocks, tuple):
            self.blocks = nn.ModuleList(blocks)
        else:
            raise ValueError(blocks.__class__)
        self.num_blocks = len(self.blocks)

    def forward(self, encoder_query, decoder_query, encoder_hidden_state, S_ns, att_mask=None, offset=0):
        eq_out = []
        dq_out = []
        S_out = []

        assert len(encoder_query.shape) == len(decoder_query.shape) == 4, f"{encoder_query.shape}, {decoder_query.shape}"
        assert encoder_query.shape[0] == decoder_query.shape[0] == self.num_blocks, self.num_blocks

        if S_ns is None:
            S_ns = [None for _ in range(self.num_blocks)]

        for B, eq, dq, ehs, s_n in zip(self.blocks, encoder_query, decoder_query, encoder_hidden_state, S_ns):
            neq, ndq, ns_n = B(eq, dq, ehs, s_n, att_mask, offset)
            eq_out.append(neq)
            dq_out.append(ndq)
            S_out.append(ns_n)

        eq_out = torch.stack(eq_out)
        dq_out = torch.stack(dq_out)
        return eq_out, dq_out, S_out


class COINSampler(nn.Module):
    def __init__(self, config, regions, sum_out=True):
        super().__init__()
        self.config = config
        self.sum_out = sum_out
        self.num_repetitions = config.num_repetitions
        self.share_S = config.share_S
        self.decoder_output = config.decoder_output
        self.layers = nn.ModuleList()
        for n in zip(*regions):
            self.layers.append(COINSamplerLayer(config, n))
        self.num_layers = len(self.layers)
        self.num_regions = len(regions)

        print(f"num layers: {self.num_layers}, num regions: {self.num_regions}")

    def forward(self, encoder_query, decoder_query, encoder_hidden_state, S_ns, att_mask=None, offset=0):
        B, T, C = encoder_query.shape
        encoder_query = encoder_query.unsqueeze(0).repeat(self.num_regions, 1, 1, 1)
        decoder_query = decoder_query.unsqueeze(0).repeat(self.num_regions, 1, 1, 1)

        if encoder_hidden_state is None:
            #encoder_hidden_state = torch.zeros(self.num_regions, B, T, C).to(encoder_query.device)
            encoder_hidden_state = [None for _ in range(self.num_regions)]
            #encoder_hidden_state = torch.randn(self.num_regions, B, T, C).to(encoder_query.device) #for _ in range(self.num_regions)]

        if self.share_S:
            s_n = S_ns
        else:
            if S_ns is None:
                S_ns = [None for _ in range(self.num_layers)]
            S_out = []

        #print(encoder_hidden_state)
        #encoder_query = encoder_hidden_state

        for _ in range(self.num_repetitions):
            for i, L in enumerate(self.layers):
                if not self.share_S:
                    s_n = S_ns[i]
                encoder_query, decoder_query, s_n = L(encoder_query, decoder_query, encoder_hidden_state, s_n, att_mask, offset)
                if not self.share_S:
                    S_out.append(s_n)
        
        S_ns = s_n if self.share_S else S_out
        encoder_hidden_state = encoder_query.clone()#.detach()

        if self.decoder_output == "strict":
            out = decoder_query
        elif self.decoder_output == "adaptive":
            out = encoder_query + decoder_query
        elif self.decoder_output == "none":
            out = encoder_query

        if self.sum_out:
            out = out.sum(0)
        return out, encoder_hidden_state, S_ns


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
    def __init__(self, config, region_config=None, sum_out=True):
        super().__init__(config)
        self.config = config
        self.disable_teacher_forcing = config.disable_teacher_forcing
        self.num_regions = config.num_regions
        if config.block_ioh_schema is None:
            config.block_ioh_schema = [config.hidden_size for _ in  range(len(config.decoder_schema) + 1)]
        
        self.regions = nn.ModuleList([
            COINRegionHolder(config) for _ in range(self.num_regions)
        ])
        self.sampler = COINSampler(config, self.regions, sum_out)

        self.encoder_embeddings = COINEmbeddings(config)
        self.decoder_embeddings = COINEmbeddings(config)
        self.pooler = COINPooler(config)
        print("num core parameters: {:,}".format(num_parameters(self)))
        self.post_init()

    def _init_S(self):
        return [
            None for _ in range(self.num_regions)
        ]

    def _sample(self, input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0):
        encoder_query = self.encoder_embeddings(input_ids, inputs_embeds)
        if self.disable_teacher_forcing or not self.training:
            decoder_input_ids = input_ids
            decoder_inputs_embeds = inputs_embeds
        decoder_query = self.decoder_embeddings(decoder_input_ids, decoder_inputs_embeds)
        return self.sampler(encoder_query, decoder_query, encoder_hidden_state, S, attention_mask, offset)
        
    def _chunk_sample(self, input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0):
        T = input_ids.shape[1]
        assert (T % self.config.chunkwise_num_chunks) == 0, f"{T} % {self.config.chunkwise_num_chunks} != 0"
        nc_step = T // self.config.chunkwise_num_chunks
        Y = []
        for j in range(0, T, nc_step):
            i_ = j + offset
            u = j + nc_step
            ii = input_ids[:, j:u] if input_ids is not None else None
            dii = decoder_input_ids[:, j:u] if decoder_input_ids is not None else None
            ie = inputs_embeds[:, j:u, :] if inputs_embeds is not None else None
            die = decoder_inputs_embeds[:, j:u, :] if decoder_inputs_embeds is not None else None
            am = attention_mask[:, j:u] if attention_mask is not None else None

            y, encoder_hidden_state, S = self._sample(ii, dii, encoder_hidden_state, S, ie, die, am, i_)
            Y.append(y)
        Y = torch.cat(Y, 1)
        return Y, encoder_hidden_state, S

    def forward(self, input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, **kwargs):
        if not self.config.print_checks:
            if self.config.forward_method == "chunkwise" and self.config.apply_chunking_globally:
                out, encoder_hidden_state, S = self._chunk_sample(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
            else:
                out, encoder_hidden_state, S = self._sample(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
        else:
            self.config.forward_method = "parallel"
            print("SAMPLE ONE CALL")
            Y_par, ehs_par, S_par = self._sample(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
            self.config.forward_method = "chunkwise"
            Y_chn, ehs_chn, S_chn = self._chunk_sample(input_ids, decoder_input_ids, encoder_hidden_state, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
            
            chrs = torch.allclose(Y_par, Y_chn, atol=1e-5)
            print("global check:", chrs, "SUCCESS" if chrs else "FAILED")
            out, encoder_hidden_state, S = Y_chn, ehs_chn, None

        pooled_out = self.pooler(out)
        return out, pooled_out, S, encoder_hidden_state

    
class COINForSequenceClassification(COINPreTrainedModel):
    def __init__(self, config, region_config=None):
        super().__init__(config)
        self.config = config
        self.coin = COINModel(config, region_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.cls_heads = nn.ModuleList([
        #    nn.Sequential(
        #        COINPooler(config),
        #        nn.Dropout(config.hidden_dropout_prob),
        #        nn.Linear(config.hidden_size, config.num_labels) 
        #    ) for _ in range(config.num_repetitions)
        #])
        self.loss_fn = nn.BCEWithLogitsLoss()#reduction="none")
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


