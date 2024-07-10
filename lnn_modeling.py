import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class Output:
    logits: torch.Tensor = None
    encoder_hidden_state: torch.Tensor = None
    S: torch.Tensor = None
    C: torch.Tensor = None
    loss: torch.Tensor = None
    aux_loss: torch.Tensor = None


class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros((units, units), dtype=np.int32)
        self.input_dim = None
        self.output_dim = None

    def is_built(self):
        return self.input_dim is not None

    def get_neurons_of_layer(self, layer_id):
        return list(range(self.units))

    def build(self, input_dim):
        if not self.input_dim is None and self.input_dim != input_dim:
            raise ValueError(f"conflicting input dimensions {self.input_dim} != {input_dim}")
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.adjacency_matrix)
    
    def sensory_erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.sensory_adjacency_matrix)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros((input_dim, self.units), dtype=np.int32)

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    def _synapse_check(self, src, dest, polarity):
        if src < 0 or src >= self.units:
            raise ValueError(f"invalid src for synapse {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"invalid dest for synapse {dest}")
        if not polarity in (1, -1):
            raise ValueError(f"invalid polarity {polarity}")

    def add_synapse(self, src, dest, polarity):
        self._synapse_check(src, dest, polarity)
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError(f"cannot add synapses before build() is called")
        self._synapse_check(src, dest, polarity)
        self.sensory_adjacency_matrix[src, dest] = polarity


class FullyConnectedWiring(Wiring):
    def __init__(self, units, output_dim=None, erev_init_seed=1111, self_connections=True):
        super().__init__(units)
        if output_dim is None:
            output_dim = units
        self.self_connections = self_connections
        self.set_output_dim(output_dim)
        self._rng = np.random.default_rng(erev_init_seed)
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
            polarity = self._rng.choice([-1, 1, 1])
            self.add_synapse(src, dest, polarity)
        
    def build(self, input_shape):
        super().build(input_shape)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_sensory_synapse(src, dest, polarity)



class CfCCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.mode = config.mode
        assert self.mode in ("default", "pure", "no_gate"), self.mode
        self.sparsity_mask = config.sparsity_mask

        backbone_activation = nn.Tanh

        self.backbone = None
        self.backbone_layers = config.backbone_layers
        if self.backbone_layers > 0:
            #print(self.input_size, config.backbone_units, self.input_size + config.backbone_units)
            layer_list = nn.ParameterList([
                #nn.Linear(self.input_size + self.hidden_size, config.backbone_units),
                nn.Linear(self.input_size + config.backbone_units, config.backbone_units),
                backbone_activation()
            ])
            for i in range(1, self.backbone_layers):
                layer_list.append(nn.Linear(config.backbone_units, config.backbone_units))
                layer_list.append(backbone_activation())
                if config.backbone_dropout > 0.0:
                    layer_list.append(nn.Dropout(config.backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(self.hidden_size + self.input_size if self.backbone_layers == 0 else config.backbone_units)

        self.ff1 = nn.Linear(cat_shape, self.hidden_size)
        if self.mode == "pure":
            self.w_tau = nn.Parameter(torch.zeros(1, self.hidden_size))
            self.A = nn.Parameter(torch.ones(1, self.hidden_size))
        else:
            self.ff2 = nn.Linear(cat_shape, self.hidden_size)
            self.time_a = nn.Linear(cat_shape, self.hidden_size)
            self.time_b = nn.Linear(cat_shape, self.hidden_size)
        
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):
        #print(input.shape, hx.shape)
        X = torch.cat((input, hx), 1)
        if self.backbone_layers > 0:
        #    print(self.backbone)
            X = self.backbone(X)
        if self.sparsity_mask is not None:
            ff1 = F.linear(X, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(X)
        if self.mode == "pure":
            # solve
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # CfC
            if self.sparsity_mask is not None:
                ff2 = F.linear(X, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(X)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            t_a = self.time_a(X)
            t_b = self.time_b(X)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        #print(new_hidden.shape)
        return new_hidden, new_hidden


class WiredCfCCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.input_size is not None:
            config.wiring.build(config.input_size)
        if not config.wiring.is_built():
            raise ValueError("wiring is not build yet")
        self._wiring = config.wiring

        self._layers = []
        in_features = config.wiring.input_dim
        for l in range(config.wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units][prev_layer_neurons, :]
            input_sparsity = np.concatenate((
                input_sparsity,
                np.ones((len(hidden_units), len(hidden_units)))
            ), axis=0)

            rnn_config = LNNConfig(
                input_size=in_features,
                hidden_size=len(hidden_units),
                mode=config.mode,
                backbone_activation="tanh",
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0.0,
                sparsity_mask=input_sparsity,
            )
            rnn_cell = CfCCell(rnn_config)
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)

    def forward(self, input, hx, timespans):
        h_state = torch.split(hx, self.layer_sizes, dim=1)
        new_h_state = []
        inputs = input

        for i in range(self.num_layers):
            h, _ = self._layers[i](inputs, h_state[i], timespans)
            inputs = h
            new_h_state.append(h)
        
        new_h_state = torch.cat(new_h_state, 1)
        return h, new_h_state


class CfC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.input_size
        self.wiring_or_units = config.units
        self.proj_size = config.proj_size
        
        self.batch_first = True
        self.return_sequence = True

        backbone_units = config.backbone_units
        if isinstance(config.units, Wiring):
            self.wired_mode = True
            if backbone_units is not None:
                raise ValueError(f"Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError(f"Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError(f"Cannot use backbone_dropout in wired mode")

            self.wiring = config.units
            self.state_size = self.wiring.units
            self.output_size = self.wiring.output_dim
            self.rnn_cell = WiredCfCCell(config)
        else:
            self.wired_false = True
            self.state_size = config.units
            self.output_size = self.state_size
            self.rnn_cell = CfCCell(config)
        self.use_mixed = config.mixed_memory
        if self.use_mixed:
            raise NotImplementedError()
        if config.proj_size is None:
            self.fc = nn.Identity()
        else:
            #self.fc = nn.Linear(self.output_size, self.proj_size)
            self.fc = nn.Linear(config.hidden_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None):
        is_batched = input.dim() == 3
        batch_dim, seq_dim = 0, 1
        if not is_batched:
            input = input.unsqueeze(0)
            if timespans is not None:
                timespans = timespans.unsqueeze(0)
        
        B, T, C = input.shape

        if hx is None:
            h_state = torch.zeros(B, self.state_size).to(input.device)
            c_state = torch.zeros(B, self.state_size).to(input.device) if self.use_mixed else None
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError("tuple (h0, c0) expected, tensor found")
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    raise RuntimeError(f"for batched 2d input, hx and cx expected to be 2d != {h_state.dim()}")
            else:
                if h_state.dim() != 1:
                    raise RuntimeError(f"for unbatched 1d input, hx and cx expected to be 1d != {h_state.dim()}")
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0)
        
        out_seq = []
        for t in range(T):
            X_t = input[:, t]
            ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            if self.use_mixed:
                raise NotImplementedError()
            h_out, h_state = self.rnn_cell(X_t, h_state, ts)
            if self.return_sequence:
                out_seq.append(self.fc(h_out))
        
        if self.return_sequence:
            readout = torch.stack(out_seq, 1)
        else:
            readout = self.fc(h_out)
        hx = (h_state, c_state) if self.use_mixed else h_state
        return readout, hx
        

class CfCForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        #self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.cfc = CfC(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S, labels, attention_mask=None, **kwargs):
        #if not self.training:
        #    decoder_input_ids = input_ids.clone()

        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        #d_X = F.one_hot(decoder_input_ids.long(), self.config.vocab_size).float()
        #d_X = self.decoder_embeddings(d_X)
        #Y, S = self.coin(X, d_X, attention_mask, S)
        Y, _ = self.cfc(X)
        #Y = Y.flip(-1)
        Y = self.classifier(Y[:, -1])

        labels = labels.long()
        #Y = Y.flip(-1)
        loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class CfCForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),
            #nn.LayerNorm(config.hidden_size, eps=config.hidden_dropout_prob),
            #nn.Dropout(config.hidden_dropout_prob)
        )
        #self.decoder_embeddings = nn.Sequential(
        #    nn.Embedding(config.vocab_size, config.hidden_size),
        #    #nn.LayerNorm(config.hidden_size, eps=config.hidden_dropout_prob),
        #    #nn.Dropout(config.hidden_dropout_prob)
        #)
        self.cfc = CfC(config)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S, attention_mask, labels, **kwargs):
        X = self.word_embeddings(input_ids)
        #d_X = self.decoder_embeddings(decoder_input_ids) if decoder_input_ids is not None else None
        #Y, S = self.coin(X, d_X, attention_mask)
        Y, _ = self.cfc(X)
        #Y = Y.flip(-1)
        Y = self.dropout(Y)
        Y = self.pooler(Y[..., 0, :])
        Y = self.classifier(Y)
        #labels = (labels == 0).long()#.flip(-1)
        #Y = (Y.argmax(-1) == 0).int()
        loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class CfCForBucketSort(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.cfc = CfC(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S=None, labels=None, **kwargs):
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)

        Y, _ = self.cfc(X)

        Y = self.lm_head(Y)

        labels = labels.long()
        loss = self.loss_fn(Y.view(-1, Y.shape[-1]), labels.view(-1))
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


@dataclass
class CfCConfig:
    input_size: int = 256
    hidden_size: int = 256
    sparsity_mask: Optional[torch.Tensor] = None
    backbone_layers: int = 1
    backbone_units: int = 128
    backbone_dropout: float = 0.0
    mode: str = "default"
    units: Union[int, Wiring] = 2
    proj_size: int = 256
    wiring: Wiring = None
    backbone_activation: str = "tanh"
    mixed_memory: bool = False
    vocab_size: int = 30522
    num_labels: int = 2
    hidden_dropout_prob: float = 0.1


class LTCCell(nn.Module):
    def __init__(self, wiring, in_features=None, input_mapping="affine", output_mapping="affine", ode_unfolds=6, epsilon=1e-8, implicit_param_constraints=False):
        super().__init__()
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        self.make_positive_fn = (
            nn.Softplus() if implicit_param_constraints else nn.Identity()
        )
        self._implicit_param_constraints = implicit_param_constraints

        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = nn.ReLU()
        self.state_size = wiring.units
        self.sensory_size = wiring.input_dim
        self.motor_size = wiring.output_dim

        self.gleak = nn.Parameter(torch.rand(self.state_size) * (1.0 - 0.001) + 0.001)
        self.vleak = nn.Parameter(torch.rand(self.state_size) * (0.2 - -0.2) + -0.2)
        self.cm = nn.Parameter(torch.rand(self.state_size) * (0.6 - 0.4) + 0.4)
        self.sigma = nn.Parameter(torch.rand(self.state_size, self.state_size) * (8 - 3) + 3)
        self.mu = nn.Parameter(torch.rand(self.state_size, self.state_size) * (0.8 - 0.3) + 0.3)
        self.w = nn.Parameter(torch.rand(self.state_size, self.state_size) * (1.0 - 0.001) + 0.001)
        self.erev = nn.Parameter(torch.Tensor(self._wiring.erev_initializer()))
        self.sensory_sigma = nn.Parameter(torch.rand(self.sensory_size, self.state_size) * (8 - 3) + 3)
        self.sensory_mu = nn.Parameter(torch.rand(self.sensory_size, self.state_size) * (0.8 - 0.3) + 0.3)
        self.sensory_w = nn.Parameter(torch.rand(self.sensory_size, self.state_size) * (1.0 - 0.001) + 0.001)
        self.sensory_erev = nn.Parameter(torch.Tensor(self._wiring.sensory_erev_initializer()))
        self.sparsity_mask = nn.Parameter(torch.abs(torch.Tensor(self._wiring.sensory_adjacency_matrix)))
        self.sensory_sparsity_mask = nn.Parameter(torch.abs(torch.Tensor(self._wiring.sensory_adjacency_matrix)))
        if self._input_mapping in ("affine", "linear"):
            self.input_w = nn.Parameter(torch.ones(self.sensory_size))
        if self._input_mapping == "affine":
            self.input_b = nn.Parameter(torch.zeros(self.sensory_size))
        if self._output_mapping in ("affine", "linear"):
            self.output_w = nn.Parameter(torch.ones(self.motor_size))
        if self._output_mapping == "affine":
            self.output_b = nn.Parameter(torch.zeros(self.motor_size))

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return F.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.make_positive_fn(
            self.sensory_w
        ) * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_w_activation = (
            sensory_w_activation * self.sensory_sparsity_mask
        )

        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.make_positive_fn(self.cm) / (
            elapsed_time / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        w_param = self.make_positive_fn(self.w)
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self.mu, self.sigma
            )

            w_activation = w_activation * self.sparsity_mask

            rev_activation = w_activation * self.erev

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self.gleak)
            numerator = cm_t * v_pre + gleak * self.vleak + w_numerator
            denominator = cm_t + gleak + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self.input_w
        if self._input_mapping == "affine":
            inputs = inputs + self.input_b
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]  # slice

        if self._output_mapping in ["affine", "linear"]:
            output = output * self.output_w
        if self._output_mapping == "affine":
            output = output + self.output_b
        return output
    
    def forward(self, inputs, states, elapsed_time=1.0):
        # Regularly sampled mode (elapsed time = 1 second)
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states, elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, next_state


@dataclass
class LTCConfig:
    input_size: int = 256
    hidden_size: int = 1024
    units: Union[int, Wiring] = 256
    mixed_memory: bool = False
    input_mapping: str = "affine"
    output_mapping: str = "affine"
    ode_unfolds: int = 6
    epsilon: float = 1e-8
    implicit_param_constraints: bool = True
    vocab_size: int = 30522
    num_labels: int = 2
    hidden_dropout_prob: float = 0.1


class LTC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.input_size
        self.wiring_or_units = config.units
        
        if isinstance(config.units, Wiring):
            wiring = config.units
        else:
            wiring = FullyConnectedWiring(config.units)
        
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=config.input_size,
            input_mapping=config.input_mapping,
            output_mapping=config.output_mapping,
            ode_unfolds=config.ode_unfolds,
            epsilon=config.epsilon,
            implicit_param_constraints=config.implicit_param_constraints,
        )
        self._wiring = wiring
        self.use_mixed = config.mixed_memory
        if self.use_mixed:
            raise NotImplementedError()
        self.state_size = self._wiring.units
        self.sensory_size = self._wiring.input_dim
        self.motor_size = self._wiring.output_dim
        self.output_size = self.motor_size

    def forward(self, input, hx=None, timespans=None):
        B, T, C = input.shape
        if hx is None:
            h_state = torch.zeros(B, self.state_size).to(input.device)
            c_state = None
        else:
            h_state, c_state = hx, None

        out_seq = []
        for t in range(T):
            inputs = input[:, t]
            ts = 1.0 if timespans is None else timespans[:, t].squeeze()

            h_out, h_state = self.rnn_cell(inputs, h_state, ts)
            out_seq.append(h_out)
        
        readout = torch.stack(out_seq, 1)
        return readout, h_state


class LTCForParityCheck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        #self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.ltc = LTC(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S, labels, attention_mask=None, **kwargs):
        #if not self.training:
        #    decoder_input_ids = input_ids.clone()

        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)
        #d_X = F.one_hot(decoder_input_ids.long(), self.config.vocab_size).float()
        #d_X = self.decoder_embeddings(d_X)
        #Y, S = self.coin(X, d_X, attention_mask, S)
        Y, _ = self.cfc(X)
        #Y = Y.flip(-1)
        Y = self.classifier(Y[:, -1])

        labels = labels.long()
        #Y = Y.flip(-1)
        loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class LTCForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_size),
            #nn.LayerNorm(config.hidden_size, eps=config.hidden_dropout_prob),
            #nn.Dropout(config.hidden_dropout_prob)
        )
        #self.decoder_embeddings = nn.Sequential(
        #    nn.Embedding(config.vocab_size, config.hidden_size),
        #    #nn.LayerNorm(config.hidden_size, eps=config.hidden_dropout_prob),
        #    #nn.Dropout(config.hidden_dropout_prob)
        #)
        self.ltc = LTC(config)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S, attention_mask, labels, **kwargs):
        X = self.word_embeddings(input_ids)
        #d_X = self.decoder_embeddings(decoder_input_ids) if decoder_input_ids is not None else None
        #Y, S = self.coin(X, d_X, attention_mask)
        Y, _ = self.ltc(X)
        #Y = Y.flip(-1)
        Y = self.dropout(Y)
        Y = self.pooler(Y[..., 0, :])
        Y = self.classifier(Y)
        #labels = (labels == 0).long()#.flip(-1)
        #Y = (Y.argmax(-1) == 0).int()
        loss = self.loss_fn(Y, labels)
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )


class LTCForBucketSort(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.decoder_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.ltc = LTC(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, S=None, labels=None, **kwargs):
        X = F.one_hot(input_ids.long(), self.config.vocab_size).float()
        X = self.embeddings(X)

        Y, _ = self.ltc(X)

        Y = self.lm_head(Y)

        labels = labels.long()
        loss = self.loss_fn(Y.view(-1, Y.shape[-1]), labels.view(-1))
        return Output(
            logits=Y,
            S=S,
            loss=loss
        )
