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


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 1536,
        rms_norm_eps: float = 1e-6,
        num_layers: int = 4,
        num_labels: int = 2,
        vocab_size: int = 5027,
        max_position_embeddings: int = 512,
        apply_causal_mask: bool = True,
        pad_token_id: int = 0,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.apply_causal_mask = apply_causal_mask
        self.initializer_range = initializer_range


class TransformerPreTrainedModel(PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "transformer"
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


class TransformerOutput:
    def __init__(self, logits=None, S=None, C=None, loss=None, aux_loss=None):
        self.logits = logits
        self.encoder_hidden_state = None
        self.S = S
        self.C = C
        self.loss = loss
        self.aux_loss = aux_loss


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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wb_Q = nn.Linear(config.hidden_size, config.hidden_size)
        self.Wb_K = nn.Linear(config.hidden_size, config.hidden_size)
        self.Wb_V = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.rope = RotaryEmbedding(config.hidden_size)
        self.head_dim = config.hidden_size
        self.apply_mask = config.apply_causal_mask

    def forward(self, x, att_mask=None):
        B, T, C = x.shape
        Q = self.Wb_Q(x)
        K = self.Wb_K(x)
        V = self.Wb_V(x)
        D = _get_D(1, T).to(x.device)

        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(K)

        A = (Q @ K.transpose(-2, -1)) / (self.head_dim ** .5)
        if self.apply_mask:
            A.masked_fill_(D == 0, float("-inf"))
        A = F.softmax(A, -1) @ V
        out = self.fc(A)
        return out


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)#, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)#, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)#, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()#ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.att = Attention(config)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.glu = GLU(config.hidden_size, config.intermediate_size, config.hidden_size)
        self.glu = LlamaMLP(config)

    def forward(self, x, att_mask):
        norm_x = self.norm(x)
        y = self.att.forward(norm_x, att_mask) + x
        #return y
        z = self.post_norm(y)
        z = self.glu(z) + y
        return z


class SkipGateLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.in_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.att = Attention(config)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.glu = LlamaMLP(config)

    def forward(self, x, att_mask):
        x = self.norm(x)
        x = self.in_proj(x)
        y = self.att(x, att_mask) + x
        z = self.post_norm(y)
        z = self.post_proj(z)
        z = self.glu(z) + z
        return z


__LAYER_CLASS__ = SkipGateLayer
#__LAYER_CLASS__ = Layer

class TransformerModel(TransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            __LAYER_CLASS__(config) for _ in range(config.num_layers)
        ])
        self.post_init()

    def forward(self, input_ids, inputs_embeds, attention_mask):
        x = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        for L in self.layers:
            x = L(x, attention_mask)
        return x


class TransformerForCausalLM(TransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self,
        input_ids, 
        inputs_embeds=None, 
        attention_mask=None, 
        labels=None, 
        num_logits_to_keep=0, 
        **kwargs
    ):
        logits = self.model(input_ids, inputs_embeds, attention_mask)
        logits = self.lm_head(logits[:, -num_logits_to_keep:, :])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return TransformerOutput(
            logits=logits,
            loss=loss
        )


class TransformerPredictionHeadTransform(nn.Module):
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


class TransformerLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TransformerPredictionHeadTransform(config)

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


class TransformerOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TransformerLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TransformerForMaskedLM(TransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.apply_causal_mask = False
        self.model = TransformerModel(config)
        self.cls = TransformerOnlyMLMHead(config)
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
        return TransformerOutput(
            logits=scores,
            loss=loss
        )
