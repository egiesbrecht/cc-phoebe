import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union, Optional, Tuple, List
from typing import overload
import numbers
import weakref
import math
import numpy as np
from rotary_embeddings import RotaryEmbedding
from contextual_position_embeddings import CoPE
import snntorch as snn


@dataclass
class RNNConfig:
    hidden_size: int = 256
    intermediate_size: int = 256
    max_position_embeddings: int = 40
    vocab_size: int = 40
    hidden_dropout_prob: float = 0.1
    num_hidden_layers: int = 1
    num_labels: int = 1
    memory_size: int = 256
    memory_n_stacks: int = 5
    group_norm_num: int = 32
    group_norm_channels: int = 256
    group_norm_eps: float = 1e-05
    layer_norm_eps: float = 1e-12
    rope_dim: int = 32
    layer_norm_eps: float = 1e-12
    carry_over_S: bool = False,
    chunk_schema: List[int] = None


@dataclass
class Output:
    logits: torch.Tensor = None
    encoder_hidden_state: torch.Tensor = None
    S: torch.Tensor = None
    C: torch.Tensor = None
    loss: torch.Tensor = None
    aux_loss: torch.Tensor = None


class StackRNNCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        I, C, M = config.hidden_size, config.intermediate_size, config.memory_size
        self.W_ih = nn.Parameter(torch.randn(I, C))
        self.W_hh = nn.Parameter(torch.randn(C, C))
        self.b_ih = nn.Parameter(torch.randn(C) / C)
        self.b_hh = nn.Parameter(torch.randn(C) / C)
        self.W_sh = nn.Parameter(torch.randn(M, C))
        self.W_y = nn.Parameter(torch.randn(C, I))
        self.W_a = nn.Parameter(torch.randn(C, 3 * M))
        self.W_n = nn.Parameter(torch.randn(C, M))
        self.M = M
        #self.core = nn.RNN(config.hidden_size * 2, config.intermediate_size, batch_first=True)
        self.L_ih = nn.Linear(config.hidden_size, config.intermediate_size)
        self.L_hh = nn.Linear(config.intermediate_size, config.intermediate_size)
        self.L_y = nn.Linear(config.intermediate_size, config.hidden_size)

    def _core(self, x_t, h_t_1):
        l_t = self.L_ih(x_t)
        i_t = self.L_hh(h_t_1)
        r_t = l_t + i_t
        h_t = F.tanh(r_t)
        y_t = self.L_y(h_t)
        return y_t, h_t

    def forward(self, x_t, h_t_1, s_t_1):
        #return *self._core(x_t, h_t_1), s_t_1

        B, T, C = s_t_1.shape
        til_h_t_1 = h_t_1 + (s_t_1[..., 0, :] @ self.W_sh)
        #til_h_t_1 = h_t_1 
        h_t = F.tanh((x_t @ self.W_ih) + self.b_ih + (til_h_t_1 @ self.W_hh) + self.b_hh)
        y_t = (h_t @ self.W_y ) # ??? if relu
        #top_stacks = s_t_1[:, 0, :].view(B, C)
        #y_t, h_t = self._core(torch.cat((x_t, top_stacks), -1), h_t_1) 
        y_t = h_t @ self.W_y
        a_t = F.softmax(h_t @ self.W_a, -1).view(B, C, 3)
        n_t = F.relu(h_t @ self.W_n)
        
        cell_tiled_stack_actions = a_t#.unsqueeze(1).repeat(1, C, 1)
        push_action = cell_tiled_stack_actions[..., 0]
        pop_action = cell_tiled_stack_actions[..., 1]
        pop_value = s_t_1[..., 1, :]
        no_op_action = cell_tiled_stack_actions[..., 2]
        no_op_value = s_t_1[..., 0, :]

        top_new_stack = push_action * n_t + pop_action * pop_value + no_op_action * no_op_value
        top_new_stack = top_new_stack.unsqueeze(1)

        stack_tiled_stack_actions = a_t.view(B, 1, C, 3).repeat(1, T-1, 1, 1)
        push_action = stack_tiled_stack_actions[..., 0]
        push_value = s_t_1[..., :-1, :]
        pop_action = stack_tiled_stack_actions[..., 1]
        pop_extra_zero = torch.zeros(B, 1, C, device=s_t_1.device)
        pop_value = torch.cat((s_t_1[..., 2:, :], pop_extra_zero), 1)
        no_op_action = stack_tiled_stack_actions[..., 2]
        no_op_value = s_t_1[..., 1:, :]

        rest_new_stack = push_action * push_value + pop_action * pop_value + no_op_action * no_op_value
        s_t = torch.cat((top_new_stack, rest_new_stack), 1)

        return y_t, h_t, s_t


class StackRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.layers = nn.ModuleList([
            StackRNNCell(config) for _ in range(self.num_layers)
        ])
        self.W_O = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))
    
    def forward(self, X, h_0=None, S=None):
        B, T, C = X.shape
        if h_0 is None:
            h_0 = torch.randn(self.num_layers, B, C).to(X.device)
        #if S is None:
        S = torch.randn(self.num_layers, B, self.config.memory_n_stacks, self.config.memory_size).to(X.device)
        Y = []
        h_t = h_0
        s_t = S
        for t in range(T):
            y_n = X[:, t, :]
            #s_t_l = s_t[l]
            new_s_t = []
            for l in range(self.num_layers):
                y_n, h_t[l], s_t_l = self.layers[l](y_n, h_t[l], s_t[l])
                #s_t[l] = s_t_i
                new_s_t.append(s_t_l)
            Y.append(y_n)
            s_t = torch.stack(new_s_t)
        #    print("STACK:", s_t.shape)
        Y = torch.stack(Y, 1)
        return Y @ self.W_O, h_t


class StackRNNForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.rnn = StackRNN(config)
        #self.rnn = nn.RNN(config.hidden_size, config.hidden_size, config.num_hidden_layers, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels, **kwargs):
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        Y, S = self.rnn(X)
        Y = self.classifier(Y[:, -1])
        loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


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


class COINBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        I, C, T = config.hidden_size, config.intermediate_size, config.max_position_embeddings
        self.C = C
        self.W_Q = nn.Parameter(torch.randn(I, C) / I)
        self.W_K = nn.Parameter(torch.randn(I, C) / I)
        self.W_V = nn.Parameter(torch.randn(I, I) / I)
        self.b_Q = nn.Parameter(torch.randn(C) / C)

        self.E = nn.Parameter(torch.randn(C, C) )#/ C)

        self.W_tape_action = nn.Parameter(torch.randn(C, 5))

        self.W_h_back = nn.Parameter(torch.randn(C, C) / C)
        self.W_chunk_select = nn.Parameter(torch.randn(C, C) / C)

        self.vocab_fc = nn.Linear(C, config.vocab_size)

        self.leaky_spike = snn.Leaky(0.5)
        self.atan = snn.LeakyParallel.ATan.apply

        self.W_chunk_mask = nn.Parameter(torch.randn(C, 2) / 2)
        self.A_mask = nn.Parameter(torch.randn(C))
        max_chunk_len = 20
        self.call_count = 0
        self.avg_chunk_sum = 0
        self.avg_chunk_len = 0
        self.W_chunk_len = nn.Parameter(torch.randn(C, max_chunk_len) / max_chunk_len)
        self.W_asg = nn.Parameter(torch.randn(C, 2) / C)
        self.W_ag = nn.Parameter(torch.randn(C, C) / C)
        self.L_ag = nn.Linear(C, C)
        self.L_hg = nn.Linear(C, C)

        self.W_z = nn.Parameter(torch.randn(C, C) / C)
        self.U_z = nn.Parameter(torch.randn(C, C) / C)
        self.b_z = nn.Parameter(torch.randn(C))

        self.W_r = nn.Parameter(torch.randn(C, C) / C)
        self.U_r = nn.Parameter(torch.randn(C, C) / C)
        self.b_r = nn.Parameter(torch.randn(C))

        self.W_h = nn.Parameter(torch.randn(C, C) / C)
        self.U_h = nn.Parameter(torch.randn(C, C) / C)
        self.b_h = nn.Parameter(torch.randn(C))
        
        self.ih = nn.Linear(C, C)
        self.hh = nn.Linear(C, C)
        self.hj = nn.Linear(C, C)
        
        self.L_Q = nn.Linear(C, C)
        self.L_K = nn.Linear(C, C)
        self.L_V = nn.Linear(C, C)
        self.W_TT = nn.Parameter(torch.randn(T, T))
        self.W_Ct = nn.Parameter(torch.randn(C, C) / C)

        max_depth = 10
        self.d_hh = nn.ModuleList([
            nn.Linear(C, C) for _ in range(max_depth)
        ])
        self.d_ih = nn.ModuleList([
            nn.Linear(C, C) for _ in range(max_depth) 
        ])

        self.L_XG = nn.Linear(C, C*3)
        self.L_HG = nn.Linear(C, C*3)
        self.L_AG = nn.Linear(C, C*3)
        self.L_CG = nn.Linear(C, C*3)

        self.h_f = nn.Linear(C, C)
        self.h_i = nn.Linear(C, C)
        self.h_o = nn.Linear(C, C)

        self.act = nn.ReLU()
        self.fc = nn.Linear(C, C)
        self.hn = nn.Linear(C, C)
        
        self.W_nQ = nn.Parameter(torch.randn(C, C) / C)
        self.W_nK = nn.Parameter(torch.randn(C, C) / C)
        self.W_nV = nn.Parameter(torch.randn(C, C) / C)

        #self.L_n_t = nn.Linear(C, C*2)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.rope = RotaryEmbedding(config.rope_dim)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.C_state = nn.Parameter(torch.randn(I))
        self.layer_norm = nn.LayerNorm(C, eps=config.layer_norm_eps)
        self.cope = CoPE(config.max_position_embeddings, C)

        self.cat_gate = nn.Parameter(torch.randn(I, I * 2) / I)
        self.W_O = nn.Parameter(torch.randn(C, C) / C)

        self.WB_A_gate = nn.Linear(I, I)
        self.WB_H_gate = nn.Linear(I, I)

        self.WB_gru_x = nn.Linear(I, I*3)
        self.WB_gru_h = nn.Linear(I, I*3)

    def forward_parallel(self, X, att_mask, S_n):
        B, T, C = X.shape

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        #Q = self.L_Q(X)
        #K = self.L_K(X)
        #V = self.L_V(X)

        #Q = self.rope.rotate_queries_or_keys(Q)
        #K = self.rope.rotate_queries_or_keys(K)

        #Q = self.act(Q)
        #K = self.act(K)
        #V = self.act(V)

        #H, S_n = self.forward_recurrent(X, S_n)

        #Q = F.tanh(Q.cumsum(1))

        #Q = self.rope.rotate_queries_or_keys(Q)
        #Q *= att_mask.unsqueeze(-1)
        A = Q @ K.transpose(-2, -1)
        A *= _get_D(0.99, T).unsqueeze(0).to(X.device)
        A = A @ V
        
        out = A 

        out = self.dropout(out)
        #out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        out = out.transpose(-2, -1).contiguous().view(B, T, C)

        #out = self.layer_norm(out + X)

        return out, S_n

    def _rec_val(self, X, h_t=None):
        B, T, C = X.shape
        if h_t is None:
            h_t = []
            h_t.append(torch.zeros(B, C, device=X.device))
        for t in range(T):
            y_t = F.tanh(X[:, t, :] + h_t[-1])
            h_t.append(y_t)
        h = torch.stack(h_t[-T:], 1)
        return h, h_t

    def forward_recurrent(self, X, att_mask, S_n):
        B, T, C = X.shape
        
        #if S_n is None:
        #S_n = torch.zeros(B, C, device=X.device)
        #C_n = torch.zeros(B, T, device=X.device)
        out = []

        Q = self.L_Q(X)
        K = self.L_K(X) #/ (self.C )
        V = self.L_V(X)
        #KV = K.transpose(-2, -1) @ V

        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(K)

        Q = self.act(Q)
        K = self.act(K)
        V = self.act(V)

        #"""
        chunk_len = 1#T // 2
        if chunk_len > T:
            chunk_len = T
        S_n = torch.zeros(B, chunk_len, C, device=X.device)
        
        for t in range(T):
            Q_t = Q[:, t:(t+chunk_len)]
            S_n = self.hh(S_n) #+ (KV[:, t, :])# / self.C)
            #S_n *= att_mask[:, t].unsqueeze(1)
            #Q_t *= att_mask[:, t].unsqueeze(1)
            S_n = F.tanh(Q_t + S_n)
            A = S_n #@ K.T @ V
            out.append(A)
        out = torch.cat(out, 1)
        #out = S_n.unsqueeze(1)
        """

        chunk_len = 1

        #Q = F.tanh(Q.cumsum(1))

        S_n = torch.zeros(B, chunk_len, C).to(X.device)

        for t in range(0, T, chunk_len):
            Q_t = Q[:, t:t+chunk_len, :] #.cumsum(1)
            S_n = self.hh(S_n) #+ (KV[:, t:t+chunk_len, :] / self.C)
            #print(S_n.shape, Q_t.shape)
            S_n += Q_t
            S_n = F.tanh(S_n)#.cumsum(1)
            out.append(S_n)
        out = torch.cat(out, 1)
        """
        #Q = Q.unsqueeze(-1).repeat(1, 1, 1, C)
        #S = self.hh(X[:, 0])
        #S = torch.zeros(B, C, device=X.device)
        #out = []
        #S = S.cumsum(1) + self.hh(S).cumsum(2)
        #for t in range(T):
        #    S = F.tanh(self.hh(S))
        #    out.append(S)

        #out = F.tanh(S)#.cumsum(1)
        #out = torch.stack(out, 1)
        #out += Q

        #out = F.tanh(self.hh(X.cumsum(1)) + Q.cumsum(1))

        #out = self.dropout(out)
        #out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(B, T, C)

        #out = self.layer_norm(out + X)
        return out, S_n

    def forward_chunkwise(self, X, S_n):
        B, T, C = X.shape

        #Q = X @ self.W_Q
        #K = X @ self.W_K
        #V = X @ self.W_V
        chunks = []

        #Q = F.tanh(Q.cumsum(1))

        chunk_len = 1
        if chunk_len > T:
            chunk_len = T

        #print(T, chunk_len)
        gammas = torch.linspace(0.96, 0.99, T // chunk_len)#.to(X.device)
        #print(gammas)
        #print(T // chunk_len)
        S_n = torch.zeros(B, C, C).to(X.device)

        #inner_chunk = Q @ K.transpose(-2, -1)
        #inner_chunk *= _get_D(0.99, T).unsqueeze(0).to(X.device)
        #inner_chunk @= V

        for g, t in zip(gammas, range(0, T, chunk_len)):
            #Q_t = Q[:, t:t+chunk_len, :]
            #K_t = K[:, t:t+chunk_len, :]
            #V_t = V[:, t:t+chunk_len, :]
            Q_t = self.L_Q(X[:, t:t+chunk_len])
            K_t = self.L_K(X[:, t:t+chunk_len])
            V_t = self.L_V(X[:, t:t+chunk_len])

            D = _get_D(g, chunk_len).to(X.device)
            
            inner_chunk = Q_t @ K_t.transpose(-2, -1)
            inner_chunk *= D.unsqueeze(0)
            inner_chunk @= V_t

            e = torch.zeros(B, chunk_len, 1).to(X.device)
            for j in range(chunk_len):
                e[:, j, :] = g ** (j + 1)
            cross_chunk = (Q_t @ S_n) * e
            #chunks.append(cross_chunk)

            #inner_chunk = Q_t @ K_t.transpose(-2, -1)
            #inner_chunk *= D.unsqueeze(0)
            #inner_chunk @= V_t

            #O_t = inner_chunk[:, t:t+chunk_len, :] + cross_chunk
            O_t = inner_chunk + cross_chunk
            O_t = F.tanh(O_t)

            chunks.append(O_t)

            S_n = ((K_t.transpose(-2, -1) @ (V_t * D[-1].view(1, chunk_len, 1))) + (S_n * (g ** chunk_len)))

        
        out = torch.cat(chunks, 1)
        return out, S_n

    def forward_auto_rec(self, X, S_n):
        B, T, C = X.shape
        
        chunk_len = 1

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        #S_n = torch.zeros(B, chunk_len, C).to(X.device)
        H = []
        #s_t = torch.zeros(B, C, C).to(X.device)
        
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        KV = K.transpose(-2, -1) @ V

        #Q = self.rope.rotate_queries_or_keys(Q)
        #K = self.rope.rotate_queries_or_keys(K)

        #Q = self.act(Q)
        #K = self.act(K)
        #V = self.act(V)

        D = _get_D(0.99, T).to(X.device)
        #"""
        chunks = []
        m_Q = []

        for t in range(0, T, chunk_len):
            Q_t, u_g = Q[:, t:t+chunk_len, :].chunk(2, -1)
            K_t = K[:, t:t+chunk_len, :]
            V_t = V[:, t:t+chunk_len, :]
            
            h_t = F.tanh(self.hh(h_t) + Q_t)

            #print(h_t.shape, KV.shape)
            #A_t = torch.bmm(h_t, KV[:, t:t+chunk_len, :])
            A_t = h_t @ K_t.transpose(-2, -1)

            A_t @= V_t
            #A_t = torch.einsum("btc, bco -> bto", h_t, KV[:, t:t+chunk_len, :])

            H.append(h_t)
            #H.append(A_t)
            #u_g = F.sigmoid(u_g)
            #H.append(u_g * F.tanh(h_t) + (1 - u_g) * A_t)


        H = torch.cat(H, 1)
        #print(H.shape)
        #A = H @ K.transpose(-2, -1)
        #A *= D.unsqueeze(0)
        #A @= V

        #A_gate = F.sigmoid(self.WB_A_gate(X))
        #H_gate = F.sigmoid(self.WB_H_gate(X))
        #out = (H * H_gate) #+ (A * A_gate)
        #out = A
        out = H


        """
        out = []
        for t in range(0, T, chunk_len):
            Q_t = Q[:, t:t+chunk_len, :]#.cumsum(1)
            S_n = self.hh(S_n) #+ (KV[:, t:t+chunk_len, :] / self.C)
            #print(S_n.shape, Q_t.shape)
            S_n += Q_t
            S_n = F.tanh(S_n)#.cumsum(1)
            out.append(S_n)
        out = torch.cat(out, 1)
        """

        #out = self.dropout(out)
        #out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(B, T, C)

        return out, S_n

    def forward_gru(self, X, S_n):
        B, T, C = X.shape

        h_x = torch.zeros(B, C).to(X.device)
        out = []
        for t in range(T):
            X_t = self.WB_gru_x(X[:, t, :])
            h_t = self.WB_gru_h(h_x)

            X_reset, X_update, X_new = X_t.chunk(3, -1)
            h_reset, h_update, h_new = h_t.chunk(3, -1)

            reset = F.sigmoid(X_reset + h_reset)
            update = F.sigmoid(X_update + h_update)
            new = F.tanh(X_new + (reset * h_new))
            h_x = update * h_x + (1 - update) * new
            #h_x = self.hh(h_x) + self.ih(X[:, t, :])
            #h_x = F.tanh(h_x)
            out.append(h_x)
        out = torch.stack(out, 1)
        return out, S_n

    def forward_chn_prl(self, X, d_X, S_n):
        B, T, C = X.shape

        #Q = X @ self.W_Q
        #K = X @ self.W_K
        #V = X @ self.W_V
        #Q = self.L_Q(X)
        #K = self.L_K(X)
        #V = self.L_V(X)

        #Q = self.rope.rotate_queries_or_keys(Q)
        #K = self.rope.rotate_queries_or_keys(K)

        #Q = F.tanh(Q.cumsum(1))

        #Q = self.act(Q)
        #K = self.act(K)
        #K /= self.C**.5
        #V = self.act(V)

        # T = 576
        chunk_len = 2 #T #// 2
        if chunk_len > T:
            chunk_len = T
        chunk_len = max(1, chunk_len)
        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        #h_t = torch.zeros(B, C).to(X.device)
        #c_t = torch.zeros(B, C, C).to(X.device)

        pad = torch.zeros(B, chunk_len-1, C).to(X.device)
        cX = torch.cat((X, pad), 1)
        cd_X = torch.cat((d_X, pad), 1)

        #A = Q @ K.transpose(-2, -1)
        #A @= V

        H = []
        gammas = torch.linspace(0.96, 0.99, (T // chunk_len) + 1)
        glob_D = _get_D(0.96, T).unsqueeze(0).to(X.device)

        #print(d_X.shape)

        for i, t in enumerate(range(0, T, chunk_len)):
            X_t = cX[:, t:t+chunk_len]
            #if self.training:
            #    d_X_t = cd_X[:, t:t+chunk_len]
            #else:
            #    d_X_t = cX[:, t:t+chunk_len]
            #d_X_t = cX[:, t-chunk_len:t]
            #Q_t = Q[:, t:t+chunk_len]
            #K_t = K[:, t:t+chunk_len]
            #V_t = V[:, t:t+chunk_len]
            #K_t = self.L_K(h_t)
            #V_t = self.L_V(h_t)
            #Q_t = self.L_Q(X_t)
            #K_t = self.L_K(X_t) #/ (chunk_len ** .5)
            #V_t = self.L_V(X_t)
            Q_t = X_t @ self.W_Q
            #if i > 0:
            #    K_t = h_t @ self.W_K
            #    V_t = h_t @ self.W_V
            #else:
            K_t = X_t @ self.W_K
            V_t = X_t @ self.W_V
            #K_t = d_X_t @ self.W_K
            #V_t = d_X_t @ self.W_V

            #Q_t = self.rope.rotate_queries_or_keys(Q_t, offset=t)
            #K_t = self.rope.rotate_queries_or_keys(K_t, offset=t)
            #Q_t = self.act(Q_t)
            #K_t = self.act(K_t)
            #V_t = self.act(V_t)


            #X_reset, X_update, X_new = self.L_XG(X_t).chunk(3, -1)
            #H_reset, H_update, H_new = self.L_HG(h_t).chunk(3, -1)
            #Q_t = X_new

            g = gammas[i]
            #D = _get_D(g, chunk_len).to(X.device)
            D = _get_D(g, chunk_len, K_t.shape[1]).to(X.device)

            #Q_t = F.tanh(Q_t.cumsum(1))
            A_t = Q_t @ K_t.transpose(-2, -1)
            #A_t = self.hh(h_t) @ K_t.transpose(-2, -1)
            #print(A_t.shape, D.shape)
            #A_t *= D.unsqueeze(0)
            #print(X.shape, A_t.shape, glob_D.shape)
            #A_t *= glob_D[:, t:t+chunk_len, t:t+chunk_len]
            #A_t = F.softmax(A_t, -1)
            A_t = A_t @ V_t
            #A_t = A_t.flip(-1)
            #A_t = A[:, t:t+chunk_len, :]
            #print(A_t.shape)
            #e = torch.zeros(B, chunk_len, 1).to(X.device)
            #for j in range(chunk_len):
            #    e[:, j, :] = g ** (j+1)
            #C_t = (Q_t @ c_t) * e

            #dc_t = K_t.transpose(-2, -1) @ (V_t * D[-1].view(1, chunk_len, 1))
            #c_t = dc_t + (c_t * (g ** chunk_len))
            #c_t = dc_t + (c_t @ self.W_Ct)


            #A_t = Q_t @ c_t
            #c_t = K.transpose(-2, -1) @ (V)

            #c_t = c_t + (K_t.transpose(-2, -1) @ V_t)
            #A_t = Q_t @ c_t

            #print(h_t.shape, A_t.shape)
            #h_reset, h_update, h_new = self.L_HG(h_t).chunk(3, -1)
            #A_reset, A_update, A_new = self.L_AG(A_t).chunk(3, -1)
            #C_reset, C_update, C_new = self.L_CG(C_t).chunk(3, -1)

            #a_gate, h_gate = (X_t @ self.cat_gate).chunk(2, -1)
            #A_t *= a_gate
            #h_t = F.tanh(A_t)
            #h_t = F.tanh(self.hh(h_t) + A_t )#+ C_t)
            h_t = F.tanh(self.hh(h_t) + A_t) #* h_gate
            #for p in range(A_t.shape[1]):
            #    A_tp = A_t[:, p, :].unsqueeze(1)
            #    print(p, chunk_len, A_tp.shape)
            #    h_t = F.tanh(self.hh(h_t) + A_tp)
            #h_t = self.hh(h_t) + F.tanh(A_t) 

            #reset_gate = F.sigmoid(X_reset + H_reset)
            #update_gate = F.sigmoid(X_update + H_update)
            #new_state = F.tanh(A_t + (reset_gate * H_new))
            #h_t = update_gate * h_t + (1 - update_gate) * new_state

            H.append(h_t)
            #H.append(A_t)

        out = torch.cat(H, 1)
        #out = A_t
        #out = torch.stack(H, 1)
        
        #out = self.dropout(out)
        #out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(B, T, C)

        #out = self.layer_norm(out + X)

        return out, S_n

    def autochunk(self, X):
        B, T, C = X.shape

        Q = self.L_Q(X)
        K = self.L_K(X)
        V = self.L_V(X)
        
        chunk_len = 1

        A = []
        for t in range(T):
            X_t = X[:, t:t+chunk_len]
            #Q_t = Q[:, t:t+chunk_len]
            #K_t = K[:, t:t+chunk_len]
            Q_t = self.L_Q(X_t)
            K_t = self.L_K(X_t)

            A_t = Q_t @ K_t.transpose(-2, -1)
            A.append(A_t)
        A = torch.cat(A, -1)
        #print(A.shape, V.shape)
        out = A @ V

        return out

    def forward_nlayer_steps(self, X, depth=1):
        B, T, C = X.shape
        if depth == 0:
            return X
        if depth == 1:
            chunk_len = T
        else:
            chunk_len = T# max(T // 2, 1)

        if chunk_len > T:
            chunk_len = T
        h_t = torch.zeros(B, C).to(X.device)
        H = []
        for i, t in enumerate(range(T)):
            X_t = X[:, t, :]
            Q_t = self.L_Q(X_t)
            K_t = self.L_K(X_t)
            V_t = self.L_V(X_t)
            A_t = Q_t @ K_t.transpose(-2, -1)
            A_t @= V_t
            h_t = F.tanh(self.hh(h_t) + A_t)
            #h_t = F.tanh(self.hh(h_t) + self.ih(X_t))
            #h_t = F.tanh(self.d_hh[depth](h_t) + self.d_ih[depth](X_t))
            
            #X_t = self.WB_gru_x(X[:, t, :])
            #h_g = self.WB_gru_h(h_t)

            #X_reset, X_update, X_new = X_t.chunk(3, -1)
            #h_reset, h_update, h_new = h_g.chunk(3, -1)

            #reset = F.sigmoid(X_reset + h_reset)
            #update = F.sigmoid(X_update + h_update)
            #new = F.tanh(X_new + (reset * h_new))
            #h_t = update * h_t + (1 - update) * new

            #print(i, i+1, (i+1) % chunk_len == 0, chunk_len, T)
            if ((i+1) % chunk_len == 0) or i == (T-1):
                #print("ADD", i, i % chunk_len)
                H.append(h_t)
                h_t = torch.zeros(B, C).to(X.device)

        #H.append(h_t)
        out = torch.stack(H, 1)
        
        #print(depth, T, out.shape)
        
        #out = h_t.unsqueeze(1)
        out = self.forward_nlayer_steps(out, depth-1)
        return out

    def _dummy_S_init(self, dev):
        return torch.zeros(1).to(dev)

    def memory_skip(self, X, S_n, att_mask, input_ids, chunk_len):
        #S_n = self._dummy_S_init(X.device)
        B, T, C = X.shape
        
        #chunk_len = 1#T // 2
        #chunk_len = X.mean(0).mean(-1).argmax(0)
        #chunk_len = X.mean(-1).argmax(-1).float().mean()
        #chunk_len = int(chunk_len)
        
        #chunk_len = input_ids.argmax(1).float().mean()
        #chunk_len = (X @ self.W_r).sum(-1).argmax(1).float().mean()
        #chunk_len = int(chunk_len.item())
        
        #print(T, chunk_len)
        #print(X[0])
        #print(input_ids)
        #print(input_ids.float().mean(0).argmax(-1))
        #chunk_len = F.softmax((X @ self.W_chunk_len).mean(0)[0], -1).argmax(-1).item()  
        #print(X.shape, chunk_len, self.W_chunk_len.shape, (X @ self.W_chunk_len).shape, (X @ self.W_chunk_len)[0].shape)
        #chunk_len = 2
        #chunk_len = T // 2
        #self.avg_chunk_sum += chunk_len
        #self.call_count += 1
        #if self.call_count % 500 == 0:
        #    print("avg chunk len:", self.avg_chunk_sum / self.call_count)

        #if chunk_len > T:
        #    chunk_len = T
        chunk_len = max(chunk_len, 1)
        chunk_len = min(chunk_len, T)

        pad_0 = torch.zeros(B, chunk_len-1, C).to(X.device)
        #pad_X = torch.cat((X, pad_0), 1)
        pad_X = X

        #aux_loss = torch.Tensor([0]).to(X.device)
        aux_loss = None
        #print(aux_loss)
        gammas = torch.linspace(0.96, 0.99, T // chunk_len + 1)#.to(X.device)

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        prev_A_state = torch.zeros(B, chunk_len, C).to(X.device)
        #all_A_state = torch.zeros(B, chunk_len, C).to(X.device)
        prev_scale = torch.cat((torch.linspace(0, -1, math.ceil(T/2)), torch.linspace(-1, 0, math.floor(T/2))))
        #prev_scale = torch.linspace(1, -1, T)
        prev_factor = prev_scale[chunk_len-1]
        #print(prev_factor)
        H = []
        for i, t in enumerate(range(0, T, chunk_len)):
            X_t = pad_X[:, t:t+chunk_len]
            Q_t = X_t @ self.W_Q + self.b_Q
            K_t = X_t @ self.W_K
            V_t = X_t @ self.W_V

            #Q_t = self.rope.rotate_queries_or_keys(Q_t, offset=t)
            #K_t = self.rope.rotate_queries_or_keys(K_t, offset=t)

            Q_t = self.act(Q_t)
            K_t = self.act(K_t)
            #V_t = self.act(V_t)

            #Q_t = self.rope.rotate_queries_or_keys(Q_t)
            #K_t = self.rope.rotate_queries_or_keys(K_t)

            #g = gammas[i]
            #D = _get_D(g, chunk_len).to(X.device)

            A_t = Q_t @ K_t.transpose(-2, -1)
            #A_t *= D.unsqueeze(0).transpose(-2, -1)
            A_t @= V_t

            #f_scores = F.softmax(A_t @ self.W_r, 1).topk(chunk_len, 1).indices
            #print(f_scores.shape, X_t.shape, A_t.shape)
            #A_f = X.gather(1, f_scores)
            #print(f_scores.shape, A_f.shape, A_t.shape)
            #A_t *= A_f

            #r_t = F.sigmoid(h_t @ self.W_r + X_t @ self.U_r + self.b_r)

            hat_h = self.hh(h_t)
            h_t = F.tanh(hat_h + A_t + (prev_A_state * prev_factor))
            #h_t = F.tanh(hat_h) + F.tanh(A_t) - F.tanh(prev_A_state) #* A_f
            #h_t = hat_h * r_t + F.tanh(A_t - prev_A_state) * (1 - r_t)
            #print((hat_h - prev_A_state).mean())
            #c_loss = (h_t - prev_A_state).sum()
            #if aux_loss is None:
            #    aux_loss = c_loss
            #else:
            #    aux_loss += c_loss
            
            prev_A_state = A_t
            #all_A_state = torch.cat((all_A_state, A_t), 1)

            #h_t = h_t.transpose(-2, -1).reshape(h_t.shape)
            H.append(h_t)

            #self.hh.weight = self.hh.weight + nn.Parameter(h_t.mean(1).sum(0))
            #self.hh.weight = torch.add(self.hh.weight, nn.Parameter(h_t.mean(1).sum(0)))
            #H.append(self.fc(h_t))
        
        H = torch.cat(H, 1)#[:, -T:]
        #H = torch.stack(H, 1)
        out = H
       
        #out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(B, T, C)
        #out = self.layer_norm(out + X)
        #print(out.shape)
        #assert X.shape == out.shape, f"{X.shape} != {out.shape}"

        #print(aux_loss)
        return out, S_n, aux_loss

    def dynamic_memory_skip(self, X, S_n):
        B, T, C = X.shape

        chunk_len = 40
        chunk_len = max(chunk_len, 1)
        chunk_len = min(chunk_len, T)

        max_chunks = T // chunk_len + 1

        pad_0 = torch.zeros(B, chunk_len-1, C).to(X.device)
        pad_X = torch.cat((X, pad_0), 1)

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        prev_A_state = torch.zeros(B, chunk_len, C).to(X.device)

        H = []
        for i in range(max_chunks):
            # select chunk between min and max value
            chunk_select = X @ self.W_chunk_select
            chunk_select = F.softmax(chunk_select, -1)
            #chunk_min = chunk_select.min(-1).indices
            #chunk_max = chunk_select.max(-1).indices
            chunk_select = chunk_select.topk(chunk_len, 1, largest=True)

            X_t = pad_X.gather(1, chunk_select.indices) 

            Q_t = X_t @ self.W_Q
            K_t = X_t @ self.W_K
            V_t = X_t @ self.W_V

            #Q_t = self.rope.rotate_queries_or_keys(Q_t)
            #K_t = self.rope.rotate_queries_or_keys(K_t)

            Q_t = self.act(Q_t)
            K_t = self.act(K_t)
            #V_t = self.act(V_t)

            #g = gammas[i]
            #D = _get_D(g, chunk_len).to(X.device)

            A_t = Q_t @ K_t.transpose(-2, -1)
            #A_t *= D.unsqueeze(0).transpose(-2, -1)
            A_t @= V_t

            #f_scores = F.softmax(A_t @ self.W_r, 1).topk(chunk_len, 1).indices
            #print(f_scores.shape, X_t.shape, A_t.shape)
            #A_f = X.gather(1, f_scores)
            #print(f_scores.shape, A_f.shape, A_t.shape)
            #A_t *= A_f

            hat_h = self.hh(h_t)
            h_t = F.tanh(hat_h + A_t )#+ prev_A_state) #* A_f
            
            prev_A_state = A_t
            #all_A_state = torch.cat((all_A_state, A_t), 1)

            #h_t = h_t.transpose(-2, -1).reshape(h_t.shape)
            H.append(h_t)

        out = torch.cat(H, 1)[:, :T]

        #out = out.transpose(-2, -1).contiguous().view(B, T, C)

        out = self.layer_norm(out + X)

        return out, S_n, None

    def tape_memory(self, X, S_n):
        B, T, C = X.shape

        chunk_len = 10

        chunk_len = max(chunk_len, 1)
        chunk_len = min(chunk_len, T)

        pad_0 = torch.zeros(B, chunk_len-1, C).to(X.device)
        pad_X = torch.cat((X, pad_0), 1)
        #pad_X = X

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        tape_len = T
        tape = torch.zeros(B, tape_len, chunk_len, C).to(X.device)

        H = []
        for i, t in enumerate(range(0, T, chunk_len)):
            X_t = pad_X[:, t:t+chunk_len]
            Q_t = X_t @ self.W_Q
            K_t = X_t @ self.W_K
            V_t = X_t @ self.W_V
            Q_t = self.act(Q_t)
            K_t = self.act(K_t)
            #V_t = self.act(V_t)
            #Q_t = self.rope.rotate_queries_or_keys(Q_t)
            #K_t = self.rope.rotate_queries_or_keys(K_t)
            #g = gammas[i]
            #D = _get_D(g, chunk_len).to(X.device)
            A_t = Q_t @ K_t.transpose(-2, -1)
            #A_t *= D.unsqueeze(0).transpose(-2, -1)
            A_t @= V_t
    
            #"""
            tape_actions = F.softmax(A_t @ self.W_tape_action, -1) # (B, T, 5)
            tape_return = []
            #for X_b, tape_actions in zip(X_t, all_tape_actions):
            tape_0 = tape[:, 0].clone()
            # (B, T) x (B, l, T, C)
            left = tape_actions[..., 3].view(B, 1, chunk_len, 1) * tape.roll(-1, 1)
            tape = tape + left
            right = tape_actions[..., 4].view(B, 1, chunk_len, 1) * tape.roll(1, 1)
            tape = tape + right
            noop = tape_actions[..., 0].view(B, 1, chunk_len, 1) * tape
            write = tape_actions[..., 2].unsqueeze(-1) * A_t
            tape[:, 0] = tape_0 + write
            read = tape_actions[..., 1].view(B, chunk_len, 1) * tape_0 # (T, C)
            #    tape_return.append(read)
            #X_tape = torch.stack(tape_return, 0)
            #"""
            hat_h = self.hh(h_t)
            h_t = F.tanh(hat_h + read) #+ prev_A_state) #* A_f
            prev_A_state = A_t
            H.append(h_t)

        H = torch.cat(H, 1)[:, :T]
        #H = torch.stack(H, 1)
        out = H
       
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(B, T, C)
        out = self.layer_norm(out + X)
        return out, S_n, None


    def attention_split(self, X, S_n):
        B, T, C = X.shape

        chunk_len = T // 2

        if chunk_len > T:
            chunk_len = T
        chunk_len = max(chunk_len, 1)

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        prev_l_t = torch.zeros(B, chunk_len, C).to(X.device)

        pad_0 = torch.zeros(B, chunk_len-1, C).to(X.device)
        pad_X = torch.cat((X, pad_0), 1)

        j_size = chunk_len // 4 #+ (chunk_len % 2)
        j_size = max(1, j_size)
        h_j = torch.zeros(B, j_size, C).to(X.device)

        H = []
        for t in range(0, T, chunk_len):
            l_t = []
            #h_j = torch.zeros(B, j_size, C).to(X.device)
            #prev_j_state = torch.zeros(B, j_size, C).to(X.device)
            for j in range(0, chunk_len, j_size):
                X_j = pad_X[:, t+j:t+j+j_size]
                Q_j = X_j @ self.W_Q
                K_j = X_j @ self.W_K
                V_j = X_j @ self.W_V

                A_j = Q_j @ K_j.transpose(-2, -1) @ V_j
                #h_j = self.hj(h_j) + A_j
                h_j = A_j #- prev_j_state
                #prev_j_state = A_j
                l_t.append(h_j)
                #h_t = F.tanh(self.hh(h_t) + A_j)
                #l_t.append(h_t)

            l_t = torch.cat(l_t, 1)[:, :chunk_len]
            h_t = F.tanh(self.hh(h_t) + l_t - prev_l_t)
            prev_l_t = l_t
            H.append(h_t)

        out = torch.cat(H, 1)[:, :T]

        assert X.shape == out.shape, f"{X.shape} != {out.shape}"

        return out, S_n, None

    def fold_attention(self, X, S_n):
        B, T, C = X.shape

        num_unfolds = 2
        if num_unfolds >= T:
            num_unfolds = max(num_unfolds, T//2)
        mask_chunks = max(1, T // 2)
        unfold_masks = [
            [1] * mask_chunks + [0] * mask_chunks,
            [0] * mask_chunks + [1] * mask_chunks
        ]

        h = torch.zeros(B, T, C).to(X.device)
        prev_A_state = torch.zeros(B, T, C).to(X.device)

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        for i in range(num_unfolds):
            mask = torch.Tensor(unfold_masks[i])[:T].unsqueeze(-1).to(X.device)
            K_i = K * mask
            V_i = V * mask
            Q_i = Q * mask
            c_i = K_i.transpose(-2, -1) @ V_i
            A_i = Q_i @ c_i
            #A_i *= mask

            hat_h = self.hh(h)
            h = F.tanh(hat_h + A_i - prev_A_state)
            prev_A_state = A_i
        
        out = h
        
        assert X.shape == out.shape, f"{X.shape} != {out.shape}"
        return out, S_n, None

    def dynamic_shift(self, X, S_n):
        B, T, C = X.shape

        def back_shift(y, n, h):
            h = h.unsqueeze(1).repeat(1, n, 1) #@ self.W_h_back
            # h * decay (eg linspace 0.96 - 0.99)
            #h *= torch.linspace(0.96, 0.99, n).view(1, n, 1).to(y.device)
            #return F.tanh(y + h)
            return y + h

        h_t = torch.zeros(B, C).to(X.device)
        len_sum = 0

        A = []
        H = []
        #next_chunk_len = F.softmax((X[0, 0] @ self.W_chunk_len), -1).argmax(-1).int().item()
        #next_chunk_len = max(1, next_chunk_len)
        #next_chunk_len = min(T, next_chunk_len)

        #self.avg_chunk_sum += next_chunk_len
        #self.call_count += 1
        #if self.call_count % 500 == 0:
        #    print("avg chunk len:", self.avg_chunk_sum / self.call_count)


        while len_sum < T:
            next_chunk_len = 40#T // 2
            #next_chunk_len = ()
            t1 = len_sum
            t2 = len_sum + next_chunk_len
            X_t = X[:, t1:t2]

            Q_t = X_t @ self.W_Q
            K_t = X_t @ self.W_K
            V_t = X_t @ self.W_V

            Q_t = self.act(Q_t)
            K_t = self.act(K_t)

            A_t = Q_t @ K_t.transpose(-2, -1)
            #A_t /= self.C
            #A_t = F.softmax(A_t, -1)
            A_t @= V_t
            #A.append(A_t)
            hat_h = self.hh(h_t)
            h_t = hat_h + A_t[:, -1]
            h_t = F.tanh(h_t)
            #local_shift = torch.cat((back_shift(A_t[:, :-1], A_t.shape[1] - 1, h_t), h_t.unsqueeze(1)), 1)
            local_shift = back_shift(A_t, A_t.shape[1], h_t)
            A.append(local_shift)
            #h_t = local_shift[:, -1]
            H.append(h_t)

            len_sum += next_chunk_len

        H = torch.stack(H, 1)
        A = torch.cat(A, 1)

        out = A
        out = out.transpose(-2, -1).contiguous().view(B, T, C)
        #out = self.layer_norm(out + X)

        return out, S_n, None

    #def argmax_chunks(self, X, S_n):
    #    ...

    def _next_n_a(self, n_t, X_t, A_t, c_t, h_t, att_mask):
        #print(att_mask[0])
        next_Q = X_t @ self.W_nQ
        next_K = X_t @ self.W_nQ
        next_V = X_t @ self.W_nQ
        
        next_Q = self.act(next_Q)
        next_K = self.act(next_K)

        #next_A = next_Q @ c_t
        next_A = (next_Q @ next_K.transpose(-2, -1)) @ next_V
        #hat_n = self.hn(n_t)
        #next_n = (hat_n + h_t)
        #next_n = hat_n + (h_t @ self.W_nQ)
        hat_h = self.hh(h_t)
        next_n = (hat_h + next_A)
        return next_n

    def next_state_prediction(self, X, S_n, att_mask, input_ids):
        B, T, C = X.shape

        chunk_len = T // 4
        n_len = chunk_len #* 2

        chunk_len = min(max(chunk_len, 1), T)

        pad_0 = torch.zeros(B, chunk_len-1, C).to(X.device)
        pad_X = torch.cat((X, pad_0), 1)
        #pad_X = X

        aux_loss = None
        #n_t = None
        #n_t = torch.zeros(B, chunk_len, C).to(X.device)
        
        prev_A_state = torch.zeros(B, chunk_len, C).to(X.device)

        input_ids = torch.cat((input_ids, torch.zeros(B, n_len, dtype=input_ids.dtype, device=input_ids.device)), 1)
        
        if self.config.carry_over_S:
            if S_n is not None:
                h_t, n_t = S_n.unbind()
            else:
                print("INIT NEW S")
                h_t = torch.zeros(B, chunk_len, C).to(X.device)
                n_t = torch.ones(B, n_len, C).to(X.device) @ self.W_nQ
        else:
            h_t = torch.zeros(B, chunk_len, C).to(X.device)
            n_t = torch.ones(B, n_len, C).to(X.device) @ self.W_nQ

        R_t = torch.zeros(B, n_len, C).to(X.device)
        #print("NEW STATE")
        H = []
        for i, t in enumerate(range(0, T, chunk_len)):
            #new_n_t = F.tanh(self.hn(n_t) + R_t)
            X_t = pad_X[:, t:t+chunk_len] #+ n_t
            #X_t = torch.cat((pad_X[:, t:t+chunk_len], n_t), 1)
            Q_t = X_t @ self.W_Q
            K_t = X_t @ self.W_K
            V_t = X_t @ self.W_V

            #Q_t = self.rope.rotate_queries_or_keys(Q_t, offset=t)
            #K_t = self.rope.rotate_queries_or_keys(K_t, offset=t)

            Q_t = self.act(Q_t)
            K_t = self.act(K_t)
            V_t = self.act(V_t)

            #Q_t = self.rope.rotate_queries_or_keys(Q_t)
            #K_t = self.rope.rotate_queries_or_keys(K_t)

            #g = gammas[i]
            g = 0.99
            D = _get_D(g, chunk_len).to(X.device)

            c_t = Q_t @ K_t.transpose(-2, -1)
            #c_t *= D.unsqueeze(0)#.transpose(-2, -1)
            A_t = c_t @ V_t
            #c_t = K_t.transpose(-2, -1) @ V_t
            #A_t = Q_t @ c_t
            #A_t = self.L_n_t(A_t.transpose(-2, -1)).transpose(-2, -1)
            #R_t = A_t[:, -n_len:]
            #A_t = A_t[:, :chunk_len]

            #new_n_t = F.tanh(self.hn(n_t) + R_t)

            hat_h = self.hh(h_t)
            h_t = F.tanh(hat_h + A_t )#- prev_A_state)

            #t_loss = (X_t[:, :chunk_len] - n_t).abs().mean()
            #n_t *= att_mask[:, t:t+chunk_len].unsqueeze(-1)#.repeat(1, 1, C)
            #xn_t = X_t * att_mask[:, t:t+chunk_len].unsqueeze(-1)
            #t_loss = self.mse_loss(n_t, X_t[:, :chunk_len])
            #print(n_t.shape, input_ids.shape)
            #t_loss = self.ce_loss(self.vocab_fc(n_t).reshape(-1, self.config.vocab_size), input_ids[:, t:t+n_len].reshape(-1))
            #print(t_loss.shape)
            #t_loss *= att_mask[:, t:t+chunk_len].unsqueeze(-1)#.repeat(1, 1, C)
            #t_loss = t_loss.mean()
            
            """
            if aux_loss is None:
                aux_loss = t_loss
            else:
                aux_loss += t_loss
            """

            #h_t = self._next_n_a(n_t, X_t, A_t, c_t, h_t, att_mask)
            #n_t = (X_t @ self.W_nQ) + h_t @ self.W_nK
            #n_t = F.tanh(self.hh(h_t) + (X_t @ self.W_nQ))
            #n_t = F.tanh(self.hh(h_t) + self.ih(n_t + A_t))
            #n_t = F.tanh(self.hn(n_t) + R_t)
            #n_t = new_n_t

            #h_t = F.softmax(torch.stack((h_t, n_t), -1), -1).sum(-1)

            #r_t = F.sigmoid(X_t @ self.W_r + self.b_r)
            #h_t = (h_t )#* r_t) #+ (n_t * (1 - r_t))

            prev_A_state = A_t
            H.append(h_t)
        
        H = torch.cat(H, 1)[:, -T:]

        #aux_loss = None
        #print(aux_loss.item())

        out = H
        #out = out.transpose(-2, -1).reshape(B, T, C)
        #out = self.layer_norm(out + X)

        S_n = h_t# torch.stack([h_t, n_t])
        return out, S_n, aux_loss

    def pseudo_next_state_pass(self, X, S_n, att_mask):
        B, T, C = X.shape

        chunk_len = T // 4

        chunk_len = min(max(chunk_len, 1), T)

        pad_0 = torch.zeros(B, chunk_len, C).to(X.device)
        pad_X = torch.cat((X, pad_0), 1)
        #pad_X = X

        prev_A_state = torch.zeros(B, chunk_len, C).to(X.device)
        
        if self.config.carry_over_S:
            if S_n is not None:
                h_t, n_t = S_n.unbind()
            else:
                print("INIT NEW S")
                next_h_t = torch.zeros(B, chunk_len, C).to(X.device)
                n_t = torch.ones(B, chunk_len, C).to(X.device) @ self.W_nQ
        else:
            next_h_t = torch.zeros(B, chunk_len, C).to(X.device)
            n_t = torch.ones(B, chunk_len, C).to(X.device) @ self.W_nQ


        def att(x):
            q = x @ self.W_Q
            k = x @ self.W_K
            v = x @ self.W_V
            q = self.act(q)
            k = self.act(k)
            v = self.act(v)
            a = q @ k.transpose(-2, -1)
            return a @ v

        def time_step(x, h):
            y = self.hh(h) + self.ih(x)
            return y
            #return F.sigmoid(y)
            #return torch.log(y)
            return F.tanh(y)

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        aux_loss = None

        X_0 = pad_X[:, 0:chunk_len]
        A_0 = att(X_0) #+ X_0
        h_t = time_step(A_0, h_t)

        H = []
        for i, t in enumerate(range(0, T, chunk_len)):
            t += chunk_len
            X_t = pad_X[:, t:t+chunk_len]
            A_t = att(X_t) #+ (h_t)

            #X_tp1 = pad_X[:, t+chunk_len : t+(chunk_len*2)]
            #A_tp1 = att(X_tp1) 

            #h_t = time_step(X_t, h_t) + time_step(X_tp1, h_t)
            #h_t = time_step(X_t, h_t) 
            #h_t = h_t + time_step(X_tp1, h_t)
            #print(A_tp1.shape, h_t.shape)
            h_tp1 = time_step(A_t, h_t) #@ self.E

            n_t = h_t + h_tp1 #+ (X_t @ self.A)
            H.append(n_t)

            h_t = time_step(A_t, h_t)
            #h_t = h_tp1

            #r_t = F.sigmoid(h_t @ self.W_r + h_tp1 @ self.U_r + self.b_r)

            #n_t = h_t + h_tp1
            #n_t = (r_t * h_t) + ((1 - r_t) * h_tp1)

            #f_loss = (h_t - ( h_tp1)).abs().mean()
            #f_loss = self.mse_loss(h_tp1, h_t)
            f_loss = None

            if aux_loss is None:
                aux_loss = f_loss
            else:
                aux_loss += f_loss

            #H.append(n_t)

            prev_A_state = A_t
            #H.append(h_t)

        H = torch.cat(H, 1)#[:, -T:]

        #aux_loss = None
        #print(aux_loss.item())

        out = H
        #out = A_0
        out = self.dropout(out)
        #out = out.transpose(-2, -1).reshape(B, T, C)
        out = self.layer_norm(out + X)

        S_n = torch.stack([next_h_t, n_t])
        return out, S_n, aux_loss

    def forward(self, X, d_X, att_mask, S_n, input_ids, chunk_len):
        #return self.forward_recurrent(X, att_mask, S_n)
        #return self.forward_parallel(X, att_mask, S_n)
        #return self.forward_chunkwise(X, S_n)
        #return self.forward_auto_rec(X, S_n)
        #return self.forward_gru(X, S_n)
        #return self.forward_chn_prl(X, d_X, S_n)
        #return self.forward_nlayer_steps(X, 2), S_n
        #return self.autochunk(X), S_n
        
        return self.memory_skip(X, S_n, att_mask, input_ids, chunk_len)
        #return self.dynamic_memory_skip(X, S_n)
        #return self.tape_memory(X, S_n)

        #return self.attention_split(X, S_n)
        #return self.fold_attention(X, S_n)
        #return self.dynamic_shift(X, S_n)

        #return self.next_state_prediction(X, S_n, att_mask, input_ids)
        #return self.pseudo_next_state_pass(X, S_n, att_mask)


class COIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_schema = config.chunk_schema
        self.layers = nn.ModuleList([
            COINBlock(config) for _ in self.chunk_schema
        ])

    def forward(self, X, d_X, att_mask, S, input_ids):
        B, T, C = X.shape
        if S is None:
            S = [None for _ in range(self.config.num_hidden_layers)]
        #    S = torch.zeros(self.config.num_hidden_layers, B, C, device=X.device)
        new_S = []
        #loss = torch.Tensor([0]).to(X.device)
        loss = None
        for L, S_n, chl in zip(self.layers, S, self.chunk_schema):
            X, ts, aux_loss = L(X, d_X, att_mask, S_n, input_ids, chl(T))
            new_S.append(ts)
            if aux_loss is not None:
                #loss += aux_loss
                if loss is None:
                    loss = aux_loss
                else:
                    loss += aux_loss
        if all([n is not None for n in new_S]):
        #if new_S is not None:
            S = torch.stack(new_S)
        #print(X.shape)
        return X, S, loss


class COINForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.coin = COIN(config)
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
        Y, S, aux_loss = self.coin(X, d_X, attention_mask, S, input_ids)
        #Y = Y.flip(-1)
        Y = self.classifier(Y[:, -1])

        labels = labels.long()
        #Y = Y.flip(-1)
        loss = self.loss_fn(Y, labels)
        if aux_loss is not None:
            loss += aux_loss#.mean()
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class COINForBucketSort(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.coin = COIN(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        #self.W_out = nn.Parameter(torch.randn())

    def forward(self, input_ids, decoder_input_ids=None, S=None, labels=None, attention_mask=None, **kwargs):
        
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        B, T, C = X.shape
        if decoder_input_ids is not None:
            d_X = F.one_hot(decoder_input_ids.long(), self.config.vocab_size).float()
            d_X = self.decoder_embeddings(d_X)
        else:
            d_X = None

        Y, S, aux_loss = self.coin(X, d_X, attention_mask, S, input_ids)

        #Y = Y[:, -math.ceil(T/2):]

        Y = self.lm_head(Y)

        #Y = Y[:, :math.ceil(T/2)]
        #Y = (Y.transpose(-2, -1) @ self.W_out).transpose(-2, -1)

        labels = labels.long()
        #print(input_ids.shape, X.shape, Y.shape, labels.shape)
        #print(input_ids, "\n", labels)
        loss = self.loss_fn(Y.reshape(-1, Y.shape[-1]), labels.view(-1))
        if aux_loss is not None:
            #print(loss.shape, aux_loss.shape)
            #print(loss, aux_loss)
            loss += aux_loss.item()
        #loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class COINForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),
            #nn.LayerNorm(config.hidden_size, eps=config.hidden_dropout_prob),
            #nn.Dropout(config.hidden_dropout_prob)
        )
        self.decoder_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),
            #nn.LayerNorm(config.hidden_size, eps=config.hidden_dropout_prob),
            #nn.Dropout(config.hidden_dropout_prob)
        )
        self.coin = COIN(config)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, decoder_input_ids, S, attention_mask, labels, **kwargs):
        X = self.word_embeddings(input_ids)
        d_X = self.decoder_embeddings(decoder_input_ids) if decoder_input_ids is not None else None
        Y, S, aux_loss = self.coin(X, d_X, attention_mask, S, input_ids)
        #Y = Y.flip(-1)
        Y = self.dropout(Y)
        Y = self.pooler(Y[..., -1, :])
        Y = self.classifier(Y)
        #labels = (labels).long()#.flip(-1)
        #Y = (Y.argmax(-1) == 0).int()
        if 1:#self.training:
            loss = self.loss_fn(Y, labels)
            if aux_loss is not None:
                loss += aux_loss
        else:
            loss = torch.autograd.Variable(aux_loss, requires_grad=True)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstms = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])

        self.exp_forget_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.exp_input_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        self.reset_parameters()

    def reset_parameters(self):
        for lstm in self.lstms:
            nn.init.xavier_uniform_(lstm.weight_ih)
            nn.init.xavier_uniform_(lstm.weight_hh)
            nn.init.zeros_(lstm.bias_ih)
            nn.init.zeros_(lstm.bias_hh)
        
        for gate in self.exp_forget_gates + self.exp_input_gates:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, input_seq, hidden_state=None):
        batch_size = input_seq.size(0)
        seq_length = input_seq.size(1)

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        output_seq = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            new_hidden_state = []
            for i, (lstm, dropout, f_gate, i_gate) in enumerate(zip(self.lstms, self.dropout_layers, self.exp_forget_gates, self.exp_input_gates)):
                if hidden_state[i][0] is None:
                    h, c = lstm(x)
                else:
                    h, c = lstm(x, (hidden_state[i][0], hidden_state[i][1]))

                f = torch.exp(f_gate(h))
                i = torch.exp(i_gate(h))
                c = f * c + i * lstm.weight_hh.new_zeros(batch_size, self.hidden_size)
                new_hidden_state.append((h, c))

                if i < self.num_layers - 1:
                    x = dropout(h)
                else:
                    x = h
            hidden_state = new_hidden_state
            output_seq.append(x)

        output_seq = torch.stack(output_seq, dim=1)
        return output_seq, hidden_state

    def init_hidden(self, batch_size):
        hidden_state = []
        for lstm in self.lstms:
            h = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
            c = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
            hidden_state.append((h, c))
        return hidden_state


class sLSTMForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.slstm = sLSTM(config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S, labels, **kwargs):
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        Y, S = self.slstm(X, S)
        Y = self.classifier(Y[:, -1])
        loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class MyRNNCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ih = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.hh = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, X, h_t):
        y = F.tanh(self.ih(X) + self.hh(h_t))
        return y


class MyRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cells = nn.ModuleList([
            MyRNNCell(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, X, h_0=None):
        #B, T, C = X.shape
        T = X.shape[-2]
        C = X.shape[-1]
        if h_0 is None:
            if X.dim() == 2:
                h_shape = (C,)
            else:
                B = X.shape[0]
                h_shape = (B, C)
            h_0 = torch.zeros(self.config.num_hidden_layers, *h_shape).to(X.device)
        h_t = h_0
        ret = []
        for t in range(T):
            cy = X[..., t, :]
            n_h_t = []
            for l in range(self.config.num_hidden_layers):
                cy = self.cells[l](cy, h_t[l])
                #h_t[l] = cy
                n_h_t.append(cy)
            ret.append(cy)
            h_t = torch.stack(n_h_t)
        ret = torch.stack(ret, 1)
        return ret, h_t


class RNNForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        #self.embeddings = nn.Linear(5, config.hidden_size)
        #self.rnn = RNN(config)
        #self.rnn = nn.RNN(config.hidden_size, config.hidden_size, config.num_hidden_layers, batch_first=True)
        #self.rnn = SimpleRNN(config.hidden_size, config.hidden_size, config.num_hidden_layers, True, config.hidden_size)
        self.rnn = MyRNN(config)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            #nn.Linear(config.hidden_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels, **kwargs):
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        #print(X.shape, self.embeddings.shape)
        X = self.embeddings(X)
        B, T, C = X.shape
        #X = X.reshape(B*T, C)
        Y, S = self.rnn(X)
        #print(Y.shape)
        #Y = Y.reshape(B, T, C)
        Y = self.classifier(Y[:, -1])
        #Y = Y[:, -1]
        #Y = self.classifier(Y)
        #print(Y)
        #print(labels)
        loss = self.loss_fn(Y, labels.long())
        #loss = self.loss_fn(Y, F.one_hot(labels, 5).float())
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )
