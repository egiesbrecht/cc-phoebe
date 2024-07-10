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


class pCOINConfig(PretrainedConfig):
    model_type = "Partial Consecutive Chain-Of-Input Network"

    def __init__(
            self,
            num_heads: int = RSIZE,
            hidden_size: int = HSIZE * RSIZE,
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
            switch_ii_decoder_ii: bool = False,
            chunk_schema: List[str] = ["T", "T // 2"],
            one_hot_encoding: bool = False,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
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
        self.switch_ii_decoder_ii = switch_ii_decoder_ii
        self.chunk_schema = chunk_schema
        self.one_hot_encoding = one_hot_encoding


class pCOINOutputClass:
    def __init__(self, logits=None, S=None, loss=None, aux_loss=None):
        self.logits = logits
        self.S = S
        self.loss = loss
        self.aux_loss = aux_loss
        self.encoder_hidden_state = None
        self.C = None


class pCOINPreTrainedModel(PreTrainedModel):
    config_class = pCOINConfig
    base_model_prefix = "pCOIN"
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


class pCOINEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, inputs_embeds=None):
        if input_ids is None and inputs_embeds is None:
            return None

        E = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds

        #emb_f = self.filter_embeddings(input_ids)
        #E *= F.softmax(emb_f, dim=1)
        X = self.norm(E)
        X = self.dropout(X)
        return X


class OneHotEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, inputs_embeds=None):
        if input_ids is None and inputs_embeds is None:
            return None
        
        if inputs_embeds is None:
            E = F.one_hot(input_ids.long(), self.config.vocab_size).float()
            E = self.embeddings(E)
        else:
            E = inputs_embeds
        
        X = E
        #X = self.norm(E)
        #X = self.dropout(X)
        return X


class pCOINBlock(nn.Module):
    def __init__(self, config, chunk_fn):
        super().__init__()
        self.config = config
        self.chunk_fn = chunk_fn
        T, C = config.max_position_embeddings, config.hidden_size
        self.W_Q = nn.Parameter(torch.randn(C, C) / C)
        self.W_K = nn.Parameter(torch.randn(C, C) / C)
        self.W_V = nn.Parameter(torch.randn(C, C) / C)
        self.bias_Q = nn.Parameter(torch.randn(C) / C)
        self.bias_K = nn.Parameter(torch.randn(C) / C)
        self.bias_V = nn.Parameter(torch.randn(C) / C)
        
        self.layer_norm = nn.LayerNorm(C, eps=config.layer_norm_eps)
        self.rope = RotaryEmbedding(config.rope_dim)
        #self.act = get_act_func(config.hidden_retention_act)
        self.act = nn.ReLU()
        self.out_act = get_act_func(config.hidden_out_act)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.retention_group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.W_h = nn.Parameter(torch.randn(C, C) / C)
        self.hh = nn.Linear(C, C)
        self.bias_h = nn.Parameter(torch.randn(C))

    def forward(self, X, dX, S_n, att_mask):
        B, T, C = X.shape
        #chunk_len = self.chunk_fn(T)
        #chunk_len = int(eval(self.chunk_fn))
        chunk_len = self.chunk_fn(T)
        chunk_len = min(T, max(chunk_len, 1))

        chunk_len = 1

        pad_0 = torch.zeros(B, chunk_len-1, C).to(X.device)
        pad_X = torch.cat((X, pad_0), 1)
        #if dX is not None:
        #    pad_dX = torch.cat((dX, pad_0), 1)
        #else:
        #    pad_dX = pad_X

        #Q = pad_X @ self.W_Q
        #K = pad_X @ self.W_K
        #V = pad_X @ self.W_V

        #Q = self.act(Q)
        #K = self.act(K)

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        prev_A_state = torch.zeros(B, chunk_len, C).to(X.device)

        A_scale = torch.cat((
            torch.linspace(0, -1, math.ceil(T / 2)),
            torch.linspace(-1, 0, math.floor(T / 2))
        ))
        A_factor = A_scale[chunk_len-1]

        H = []
        for t in range(0, T, chunk_len):
            #Q_t = Q[:, t:t+chunk_len, :]
            #K_t = K[:, t:t+chunk_len, :]
            #V_t = V[:, t:t+chunk_len, :]
            X_t = pad_X[:, t:t+chunk_len]
            #dX_t = pad_dX[:, t:t+chunk_len]
            Q_t = X_t @ self.W_Q #+ self.bias_Q
            K_t = X_t @ self.W_K #+ self.bias_K
            V_t = X_t @ self.W_V #+ self.bias_V

            Q_t = self.act(Q_t)
            K_t = self.act(K_t)

            c_t = Q_t @ K_t.transpose(-2, -1)
            A_t = c_t @ V_t

            #hat_h = h_t @ self.W_h + self.bias_h
            hat_h = self.hh(h_t)
            h_t = F.tanh(hat_h + A_t + (prev_A_state * A_factor))

            prev_A_state = A_t
            H.append(h_t)
        
        out = torch.cat(H, 1)

        return out, S_n


class pCOIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            pCOINBlock(config, f) for f in config.chunk_schema
        ])
        self.num_layers = len(self.layers)
        print(self.num_layers, "layers")

    def forward(self, X, dX, S, att_mask):
        if S is None:
            S = [None for _ in range(self.num_layers)]

        new_S = []
        for L, S_n in zip(self.layers, S):
            X, ts = L(X, None, S_n, att_mask)
            new_S.append(ts)
        if all([n is not None for n in new_S]):
            S = torch.stack(new_S)
        return X, S


class pCOINPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.act = nn.Tanh()
        self.act = nn.ReLU()

    def forward(self, X):
        X = X[:, -1, :]
        #X = self.dense(X)
        #X = self.act(X)
        return X


class pCOINModel(pCOINPreTrainedModel):
    def __init__(self, config,):
        super().__init__(config)
        self.config = config
        self.one_hot = config.one_hot_encoding
        self.encoder_embeddings = OneHotEmbeddings(config) if self.one_hot else pCOINEmbeddings(config)
        self.decoder_embeddings = OneHotEmbeddings(config) if self.one_hot else pCOINEmbeddings(config)
        self.pc = pCOIN(config)
        self.pooler = pCOINPooler(config)
        self.post_init()

    def forward(self, input_ids, decoder_input_ids, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, **kwargs):
        if self.config.disable_teacher_forcing or not self.training:
            if self.config.switch_ii_decoder_ii:
                input_ids = decoder_input_ids
                inputs_embeds = decoder_inputs_embeds
            else:
                decoder_input_ids = input_ids
                decoder_inputs_embeds = inputs_embeds
        X = self.encoder_embeddings(input_ids, inputs_embeds)
        dX = self.decoder_embeddings(decoder_input_ids, decoder_inputs_embeds)
        out, S = self.pc(X, dX, S, attention_mask)
        pooled_out = self.pooler(out)
        return out, pooled_out, S


class pCOINForSequenceClassification(pCOINPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pc = pCOINModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss() if self.config.num_labels <= 2 else nn.BCEWithLogitsLoss()
        self.post_init()
    
    def forward(self, input_ids, decoder_input_ids, S, inputs_embeds=None, decoder_inputs_embeds=None, attention_mask=None, offset=0, labels=None, **kwargs):
        _, logits, S = self.pc(input_ids, decoder_input_ids, S, inputs_embeds, decoder_inputs_embeds, attention_mask, offset)
        #logits = self.dropout(pooled_out)
        logits = self.classifier(logits)
        loss = self.loss_fn(logits, labels)# if labels is not None else None
        return pCOINOutputClass(
            logits,
            S,
            loss
        )


class pCOINForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.coin = pCOIN(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, decoder_input_ids=None, S=None, labels=None, attention_mask=None, **kwargs):
        #if not self.training:
        #    decoder_input_ids = input_ids.clone()
        
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        if decoder_input_ids is not None:
            d_X = F.one_hot(decoder_input_ids.long(), self.config.vocab_size).float()
            d_X = self.decoder_embeddings(d_X)
        else:
            d_X = None
        Y, S = self.coin(X, d_X, S, attention_mask)
        #Y = Y.flip(-1)
        Y = self.classifier(Y[:, -1])

        labels = labels.long()
        #Y = Y.flip(-1)
        loss = self.loss_fn(Y, labels)
        #if aux_loss is not None:
        #    loss += aux_loss#.mean()
        return pCOINOutputClass(
            logits=Y,
            S=S,
            loss=loss
        )