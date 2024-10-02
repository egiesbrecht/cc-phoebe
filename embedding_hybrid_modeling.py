import torch
from torch import nn
import transformers
from transformers import PreTrainedModel, PretrainedConfig, DebertaModel, DebertaForSequenceClassification, DebertaConfig
import random
from typing import Tuple, Union, Dict, List, Optional
import numpy as np


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
    if name == "none" or name is None:
        return None
    else:
        raise ValueError(f"unknown activation function '{name}'")


class HybridOutput:
    def __init__(self, logits=None, labels=None, loss=None):
        self.logits = logits
        self.encoder_hidden_state = None
        self.S = None
        self.C = None
        self.labels = labels
        self.loss = loss
        self.aux_loss = None


class HybridPreTrainedModel(PreTrainedModel):
    config_class = DebertaConfig
    base_model_prefix = "HybridEmbeddingModel"
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


def _sample_definitions(
    text,
    tokenizer, 
    incontext_dict,
    num_pos_examples,
    num_neg_examples
):
    definitions, used_context_keys = [], []
    for tok in text.split():
        if tok in incontext_dict:
            used_context_keys.append(tok)
            if incontext_dict[tok] is not None:
                definitions.append(incontext_dict[tok])

    neg_examples_keys = [n for n in incontext_dict.keys() if n not in used_context_keys]
    neg_examples, neg_exp_idcs = [], []
    for _ in range(num_neg_examples):
        val = None
        while val is None:
            key = random.choice(neg_examples_keys)
            val = incontext_dict[key]
        val = tokenizer(val)["input_ids"]
        neg_examples += val
        neg_exp_idcs.append(len(val))
    
    if len(definitions) > 1:
        pos_examples, pos_exp_idcs = [], []
        for val in random.choices(definitions, k=num_pos_examples):
            val = tokenizer(val)["input_ids"]
            pos_examples += val
            pos_exp_idcs.append(len(val))
    else:
        pos_examples = [0 for _ in range(num_pos_examples)]
        pos_exp_idcs = [1 for _ in range(num_pos_examples)]

    return (
        definitions, 
        used_context_keys, 
        pos_examples, 
        pos_exp_idcs,
        neg_examples,
        neg_exp_idcs
    )


def preprocess_for_hybrid_embeddings(
    dataset,
    tokenizer,
    labels,
    max_length,
    incontext_dict,
    num_proc=4,
    remove_columns=None,
    text_field="text", 
    num_pos_examples=5,
    num_neg_examples=5
):
    """
    simplified, unbatched version of 'model_training.preprocess_with_given_labels'
    """
    def proc(examples):
        text = examples[text_field]

        definitions, used_context_keys, pos_examples, pos_exp_idcs, neg_examples, neg_exp_idcs = _sample_definitions(
            text, tokenizer, incontext_dict, num_pos_examples, num_neg_examples
        )
        
        # padding
        pos_pad = [0 for _ in range(max_length - len(pos_examples))]
        neg_pad = [0 for _ in range(max_length - len(neg_examples))]
        pos_idcs_pad = [0 for _ in range(max_length - len(pos_exp_idcs))]
        neg_idcs_pad = [0 for _ in range(max_length - len(neg_exp_idcs))]
        pos_examples += pos_pad
        neg_examples += neg_pad
        pos_exp_idcs += pos_idcs_pad
        neg_exp_idcs += neg_idcs_pad

        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        encoding["pos_input_ids"] = pos_examples
        encoding["neg_input_ids"] = neg_examples
        encoding["pos_idcs"] = pos_exp_idcs
        encoding["neg_idcs"] = neg_exp_idcs
        return encoding

    return dataset.map(proc, batched=False, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_inclusive_embeddings(
    dataset,
    tokenizer,
    labels,
    max_length,
    incontext_dict,
    num_proc=4,
    remove_columns=None,
    text_field="text", 
    num_pos_examples=5,
    num_neg_examples=5
):
    def proc(examples):
        text = examples[text_field]

        definitions, used_context_keys, pos_examples, pos_exp_idcs, neg_examples, neg_exp_idcs = _sample_definitions(
            text, tokenizer, incontext_dict, num_pos_examples, num_neg_examples
        )

        encoding = tokenizer(text)

        input_ids = encoding["input_ids"] 


class HybridCLSHead(nn.Module):
    def __init__(self, hidden_size, num_definitions):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)
        self.num_definitions = num_definitions
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        label_states = self.fc(hidden_states[:, :self.num_definitions]).squeeze(-1)
        loss = self.loss_fn(label_states, labels)
        return label_states, loss


class HybridEmbeddingModel(nn.Module):
    def __init__(
        self,
        classifier_config=None,
        total_num_examples=10,
    ):
        super().__init__()
        self.total_num_examples = total_num_examples
        self.embedding_model = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.classifier_model = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.head = HybridCLSHead(self.classifier_model.config.hidden_size, total_num_examples)

    def save_pretrained(self, *args, **kwargs):
        return self.classifier_model.save_pretrained(*args, **kwargs)

    def from_pretrained(self, *args, **kwargs):
        return self.classifier_model,from_pretrained(*args, **kwargs)

    def _read_indexed_tensor(self, t, idcs):
        assert t.shape[0] == idcs.shape[0] == 1, "only batch size of 1 supported"
        idcs = idcs[0]
        t = t[0]
        ret = []
        l = 0
        for i in range(t.shape[0]):
            p = idcs[i].item()
            if p == 0:
                break
            y = t[l : p+l]
            ret.append(y)
            l += p
        return ret

    def _embed(self, xs):
        ret = []
        for x in xs:
            t = torch.Tensor(x).unsqueeze(0)
            emb = self.embedding_model.embeddings(t)
            y = emb[:, 0] # CLS token
            ret.append(y)
        return torch.cat(ret, 0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_input_ids: torch.Tensor,
        neg_input_ids: torch.Tensor,
        pos_idcs: torch.Tensor,
        neg_idcs: torch.Tensor,
        **kwargs
    ):
        input_embedding = self.classifier_model.embeddings(input_ids=input_ids)
        
        pos_examples = self._read_indexed_tensor(pos_input_ids, pos_idcs)
        neg_examples = self._read_indexed_tensor(neg_input_ids, neg_idcs)
        
        all_examples = pos_examples + neg_examples
        labels = [1 for _ in range(len(pos_examples))] + [0 for _ in range(len(neg_examples))]
        labeled_examples = list(zip(all_examples, labels))
        random.shuffle(labeled_examples)
        all_examples, labels = list(zip(*labeled_examples))

        definition_embedding = self._embed(all_examples)
        
        inputs_embeds = torch.cat((input_embedding, definition_embedding), 1).to(input_ids.device)
        inputs_embeds = input_embedding
        labels = torch.Tensor(labels).to(input_ids.device).unsqueeze(0)
        
        outputs = self.classifier_model(inputs_embeds=inputs_embeds).last_hidden_state
        logits, loss = self.head(outputs, labels)
        return HybridOutput(
            logits=logits,
            labels=labels,
            loss=loss
        )
