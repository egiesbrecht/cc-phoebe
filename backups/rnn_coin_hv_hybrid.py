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
        return *self._core(x_t, h_t_1), s_t_1
        

        B, T, C = s_t_1.shape
        til_h_t_1 = h_t_1 + (s_t_1[..., 0, :] @ self.W_sh)
        #til_h_t_1 = h_t_1 
        h_t = F.tanh((x_t @ self.W_ih) + self.b_ih + (til_h_t_1 @ self.W_hh) + self.b_hh)
        y_t = F.relu(h_t @ self.W_y ) # ??? if relu
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
        I, C = config.hidden_size, config.intermediate_size
        self.C = C
        self.W_Q = nn.Parameter(torch.randn(I, C) / I)
        self.W_K = nn.Parameter(torch.randn(I, C) / I)
        self.W_V = nn.Parameter(torch.randn(I, I) / I)
        
        self.ih = nn.Linear(C, C)
        self.hh = nn.Linear(C, C)
        
        self.h_q = nn.Linear(C, C)
        self.h_k = nn.Linear(C, C)
        self.h_v = nn.Linear(C, C)

        self.h_f = nn.Linear(C, C)
        self.h_i = nn.Linear(C, C)
        self.h_o = nn.Linear(C, C)

        self.act = nn.ReLU()

        self.rope = RotaryEmbedding(config.rope_dim)
        self.group_norm = nn.GroupNorm(config.group_norm_num, config.group_norm_channels,
                                       eps=config.group_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.C_state = nn.Parameter(torch.randn(I))
        self.layer_norm = nn.LayerNorm(C, eps=config.layer_norm_eps)
        self.cope = CoPE(config.max_position_embeddings, C)

        self.cat_gate = nn.Parameter(torch.randn(I, I * 2) / I)
        self.W_O = nn.Parameter(torch.randn(I * 2, I) / I * 2)

    def forward_parallel(self, X, S_n):
        B, T, C = X.shape

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(K)

        Q = self.act(Q)
        K = self.act(K)
        V = self.act(V)

        #H, S_n = self.forward_recurrent(X, S_n)

        #Q = F.tanh(Q.cumsum(1))

        #Q = self.rope.rotate_queries_or_keys(Q)

        A = Q @ K.transpose(-2, -1)
        #A *= _get_D(0.99, T).unsqueeze(0).to(X.device)
        #A = A.flip(1).cumsum(1).flip(1)
        #out = (out + (A @ V)) #/ self.C 
        #A = F.softmax(A, -1)
        A = A @ V
        #out = F.softmax(out, 2)
        out = A
        #cg = X @ self.cat_gate
        #out = (torch.cat((H, A), -1) * F.softmax(cg, -1)) @ self.W_O


        #out = self.dropout(out)
        #out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        #out = out.transpose(-2, -1).contiguous().view(B, T, C)

        out = self.layer_norm(out + X)

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

    def forward_recurrent(self, X, S_n):
        B, T, C = X.shape
        
        #if S_n is None:
        S_n = torch.zeros(B, C, device=X.device)
        C_n = torch.zeros(B, T, device=X.device)
        out = []

        Q = self.h_q(X)
        K = self.h_k(X) #/ (self.C )
        V = self.h_v(X)
        KV = K.transpose(-2, -1) @ V

        """
        for t in range(T):
            Q_t = Q[:, t, :]
            S_n = self.hh(S_n) + (KV[:, t, :])# / self.C)
            S_n = F.tanh(Q_t + S_n)
            A = S_n #@ K.T @ V
            out.append(A)
        out = torch.stack(out, 1)
        """

        chunk_len = 1

        #Q = F.tanh(Q.cumsum(1))

        S_n = torch.zeros(B, chunk_len, C).to(X.device)

        for t in range(0, T, chunk_len):
            Q_t = Q[:, t:t+chunk_len, :].cumsum(1)
            S_n = self.hh(S_n) #+ (KV[:, t:t+chunk_len, :] / self.C)
            #print(S_n.shape, Q_t.shape)
            S_n += Q_t
            S_n = F.tanh(S_n)#.cumsum(1)
            out.append(S_n)
        out = torch.cat(out, 1)

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

        #out = self.layer_norm(out + X)
        return out, S_n

    def forward_chunkwise(self, X, S_n):
        B, T, C = X.shape

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
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

        inner_chunk = Q @ K.transpose(-2, -1)
        inner_chunk *= _get_D(0.99, T).unsqueeze(0).to(X.device)
        inner_chunk @= V

        for g, t in zip(gammas, range(0, T, chunk_len)):
            Q_t = Q[:, t:t+chunk_len, :]
            K_t = K[:, t:t+chunk_len, :]
            V_t = V[:, t:t+chunk_len, :]
            D = _get_D(g, chunk_len).to(X.device)
            
            e = torch.zeros(B, chunk_len, 1).to(X.device)
            for j in range(chunk_len):
                e[:, j, :] = g ** (j + 1)
            cross_chunk = (Q_t @ S_n) * e
            #chunks.append(cross_chunk)

            #inner_chunk = Q_t @ K_t.transpose(-2, -1)
            #inner_chunk *= D.unsqueeze(0)
            #inner_chunk @= V_t

            chunks.append(inner_chunk[:, t:t+chunk_len, :] + cross_chunk)

            S_n = ((K_t.transpose(-2, -1) @ (V_t * D[-1].view(1, chunk_len, 1))) + (S_n * (g ** chunk_len)))

        
        out = torch.cat(chunks, 1)
        return out, S_n

    def forward_auto_rec(self, X, S_n):
        B, T, C = X.shape
        
        chunk_len = T // 1

        h_t = torch.zeros(B, chunk_len, C).to(X.device)
        #S_n = torch.zeros(B, chunk_len, C).to(X.device)
        H = []
        #s_t = torch.zeros(B, C, C).to(X.device)
        
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q = self.rope.rotate_queries_or_keys(Q)
        K = self.rope.rotate_queries_or_keys(K)

        Q = self.act(Q)
        K = self.act(K)
        V = self.act(V)

        D = _get_D(0.99, T).to(X.device)
        #"""
        chunks = []
        m_Q = []

        for t in range(0, T, chunk_len):
            Q_t = Q[:, t:t+chunk_len, :]#.squeeze(1)
            
            h_t = F.tanh(self.hh(h_t) + Q_t)
            H.append(h_t)

        H = torch.cat(H, 1)
        A = H @ K.transpose(-2, -1)
        A *= D.unsqueeze(0)

        A @= V

        #out = A
        out = H 

        """
        out = []
        for t in range(0, T, chunk_len):
            Q_t = Q[:, t:t+chunk_len, :].cumsum(1)
            S_n = self.hh(S_n) #+ (KV[:, t:t+chunk_len, :] / self.C)
            #print(S_n.shape, Q_t.shape)
            S_n += Q_t
            S_n = F.tanh(S_n)#.cumsum(1)
            out.append(S_n)
        out = torch.cat(out, 1)
        """

        out = self.dropout(out)
        out = self.group_norm(out.reshape(-1, self.config.group_norm_channels)).reshape(out.shape)
        out = out.transpose(-2, -1).contiguous().view(B, T, C)

        return out, S_n

    def forward(self, X, S_n):
        #return self.forward_recurrent(X, S_n)
        #return self.forward_parallel(X, S_n)
        #return self.forward_chunkwise(X, S_n)
        return self.forward_auto_rec(X, S_n)


class COIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            COINBlock(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, X, S=None):
        B, T, C = X.shape
        if S is None:
            S = torch.zeros(self.config.num_hidden_layers, B, C, device=X.device)
        new_S = []
        for L, S_n in zip(self.layers, S):
            X, ts = L(X, S_n)
            new_S.append(ts)
        S = torch.stack(new_S)
        return X, S


class COINForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.coin = COIN(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S, labels, **kwargs):
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        Y, S = self.coin(X, S)
        Y = self.classifier(Y[:, -1])
        loss = self.loss_fn(Y, labels)
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
            nn.LayerNorm(config.hidden_size, eps=config.hidden_dropout_prob),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.coin = COIN(config)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, **kwargs):
        X = self.word_embeddings(input_ids)
        Y, S = self.coin(X)
        Y = self.pooler(Y[..., -1, :])
        Y = self.classifier(Y)
        loss = self.loss_fn(Y, labels)
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
        B, T, C = X.shape
        if h_0 is None:
            h_0 = torch.zeros(self.config.num_hidden_layers, B, C).to(X.device)
        h_t = h_0
        ret = []
        for t in range(T):
            cy = X[:, t, :]
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
        #self.rnn = RNN(config)
        #self.rnn = TorchRNN(config.hidden_size, config.hidden_size, config.num_hidden_layers, batch_first=True)
        #self.rnn = SimpleRNN(config.hidden_size, config.hidden_size, config.num_hidden_layers, True, config.hidden_size)
        self.rnn = MyRNN(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels, **kwargs):
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        Y, S = self.rnn(X)
        #print(Y.shape)
        Y = self.classifier(Y[:, -1])
        #Y = self.classifier(Y)
        loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )
