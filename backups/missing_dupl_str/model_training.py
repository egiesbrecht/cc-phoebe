import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import time
import datetime
import os
import json
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union, Callable
from epoch_stats import EpochStats
import numpy as np
from tqdm import tqdm
import random
from datasets import Dataset
from transformers.utils import is_torch_fx_proxy


def _doc_pad(examples, max_length, pad_token=0, doc_token=4):
    ret = {}
    for k, v in examples.items():
        cl = []
        for n in v:
            nfac = 1
            if len(n) + 1 > max_length:
                n = n[:max_length - nfac]
            #cen = [doc_token] + n + ([pad_token] * (max_length - len(n) - nfac))
            cen = n + ([pad_token] * (max_length - len(n) - nfac)) + ([doc_token] * nfac)
            if len(cen) != max_length:
                print(len(cen), len(n), max_length - len(n) - nfac)
            cl.append(cen)
        #print([len(c) for c in cl])
        ret[k] = cl
    return ret


def preprocess_with_given_labels(
    dataset, 
    tokenizer, 
    labels, 
    label2id, 
    max_length, 
    one_label_only, 
    num_proc=4, 
    remove_columns=None, 
    text_field="text", 
    default_teacher_forcing=True,
    #teacher_forcing_prefix="The correct label is "
    teacher_forcing_prefix="",
    doc_pad_tokens=False,
):
    def proc(examples):
        text = examples[text_field]
        if doc_pad_tokens:
            encoding = tokenizer(text)
            encoding = _doc_pad(encoding, max_length)
        else:
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        if one_label_only:
            encoding["labels"] = [label2id[n] for n in examples["label"]]
        else:
            labels_batch = {k: v for k, v in examples.items() if k in labels}
            labels_matrix = np.zeros((len(text), len(labels)))
            for idx, label in enumerate(labels):
                labels_matrix[:, idx] = labels_batch[label]
            encoding["labels"] = labels_matrix.tolist()
        if default_teacher_forcing:
            encoding["decoder_input_ids"] = tokenizer([teacher_forcing_prefix + str(n) for n in encoding["labels"]], padding="max_length", truncation=True, max_length=max_length)["input_ids"]
        else:
            encoding["decoder_input_ids"] = encoding["input_ids"].copy()
        return encoding

    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def _group_examples(examples, block_size, sparsify=True, pad_token=0, doc_token=4, udoc_token=5, prefix_tokens=None, num_sparse_token=1):
    em_prefix_tokens = prefix_tokens is None
    if em_prefix_tokens:
        prefix_tokens = {}
    concatenated_examples = {}
    for k, v in examples.items():
        cl = []
        for n in v:
            cl += n
            if sparsify:
                cl += [doc_token] * num_sparse_token
        concatenated_examples[k] = cl
        if em_prefix_tokens:
            prefix_tokens[k] = []
    
    prefix_len = 0
    for k, t in prefix_tokens.items():
        if prefix_len > 0:
            assert prefix_len == len(t), f"{prefix_len} {len(t)}"
        prefix_len = len(t)
    block_size -= prefix_len

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [prefix_tokens[k] + t[i : i + block_size] for i in range(0, total_length, block_size)]
        #k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def _zip_examples(examples, block_size, src_key, tgt_key, sparsify=True, pad_token=0, doc_token=4, udoc_token=5, prefix_tokens=None, num_sparse_token=1):
    def pad_cat(x, prtok):
        prefix_len = len(prtok)
        ret = []
        cret = []
        chunk_overflow_count = 0
        for xi, p in zip(x, pos_schema):
            assert len(xi) < p, f"{len(xi)}, {p}"
            dxi = prtok + xi + ([pad_token] * (p - len(xi) - num_sparse_token - prefix_len)) + ([doc_token] * num_sparse_token)
            assert len(dxi) == p, f"{len(dxi)}, {p}\n{dxi}"
            if (len(cret) + len(dxi)) >= block_size:
                cret += [pad_token] * (block_size - len(cret))
                if len(cret) > block_size:
                #    print(f"WARNING: length of current chunk ({len(cret)}) is greater than lentgh of blocks ({block_size})")
                    cret = cret[:block_size]
                    chunk_overflow_count += 1
                assert len(cret) == block_size, f"{len(cret)}, {block_size}"
                ret.append(cret)
                cret = []
            cret += dxi
        cret += [pad_token] * (block_size - len(cret))
        ret.append(cret)
        if chunk_overflow_count > 0:
            print(f"WARNING: block size was exceeded {chunk_overflow_count} times and had to be shorted")
        return ret
    
    em_prefix_tokens = prefix_tokens is None
    if em_prefix_tokens:
        prefix_tokens = {
            k: [] for k in examples.keys()
        }

    src_szs = [len(n) for n in examples[src_key]]
    tgt_szs = [len(n) for n in examples[tgt_key]]
    pos_schema = [max(s, t)+1 for s, t in zip(src_szs, tgt_szs)]
    result = {
        k: pad_cat(v, prefix_tokens[k]) for k, v in examples.items()
    }
    return result
        

def _space_input_ids(encoding, max_length, pad_token_id):
    nenc = {}
    for k, v in encoding.items():
        ne = []
        for n in v:
            space_idcs = np.linspace(0, max_length-1, len(n), dtype=int)
            space_vals = [pad_token_id] * max_length
            for i, e in zip(space_idcs, n):
                space_vals[i] = e
            ne.append(space_vals)
        nenc[k] = ne
    return nenc


def _nenc_mpe_check(nenc, max_length):
    for k, v in nenc.items():
        for vi in v:
            if len(vi) != max_length:
                print("INVALID SHAPE", len(vi), "IN", k)
                raise ValueError(f"invalid shape {len(vi)} in {k}")


def preprocess_for_binary_sparselm(dataset, tokenizer, max_length, num_proc=4, remove_columns=None, text_field="text", pad_token_id=None):
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    
    def proc(examples):
        text = examples[text_field]
        encoding = tokenizer(text)
        nenc = _space_input_ids(encoding, max_length, pad_token_id)
        nenc["labels"] = [[float(m == 0) for m in n] for n in nenc["input_ids"]] 
        _nenc_mpe_check(nenc, max_length)
        return nenc

    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_sparselm(dataset, tokenizer, max_length, num_proc=4, remove_columns=None, text_field="text", pad_token_id=None):
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id

    mask_token = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    to_mask = .15
    chance_rand_token = .2

    def proc(examples):
        text = examples[text_field]
        encoding = tokenizer(text)
        nenc = _space_input_ids(encoding, max_length, pad_token_id)
        pad_ids = []
        for n in encoding["input_ids"]:
            cn = n 
            if (max_length - len(n)) > 0:
                cn += ([pad_token_id] * (max_length - len(n))) #+ cn
            else:
                cn = cn[:max_length]
                #cn = cn[len(n) - max_length:]
            pad_ids.append(cn)
        nenc["labels"] = pad_ids
        """nenc["labels"] = mask_tokens(
                #input_ids=pad_ids, 
                input_ids=nenc["input_ids"], 
                vocab_size=vocab_size, 
                to_mask=to_mask, 
                mask_token=mask_token, 
                return_mask_only_decoder_ids=False, 
                chance_rand_token=chance_rand_token,
                ignore_up_to=mask_token
            )"""
        _nenc_mpe_check(nenc, max_length)
        return nenc

    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_cot(dataset, tokenizer, max_length, num_proc=4, remove_columns=None, group_texts=True, text_field="text", target_field="labels", doc_token=4, pad_token=None, sparsify=False, prefix=None):
    vocab_size = tokenizer.vocab_size
    if pad_token is None:
        pad_token = tokenizer.pad_token_id
    
    def proc(examples):
        text = examples[text_field]
        if group_texts:
            encoding = tokenizer(text)
            #encoding = tokenizer([" ".join(x) for x in text])
        else:
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length) # try max_length=512
        
    if group_texts:
        encoding = _group_examples(encoding, max_length, sparsify, pad_token=pad_token, prefix_tokens=prefix_tokens)
        
    assert target_field in encoding
    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def one_hot(labels, num_labels):
    ret = []
    for l in labels:
        c = [float(0) for _ in range(num_labels)]
        #c = [0.] * num_labels
        c[l] = 1.
        ret.append(c)
    return ret


def preprocess_for_seq2seq_swag(
    dataset,
    tokenizer,
    max_length,
    num_proc=4,
    remove_columns=None,
    ctx_field="ctx",
    endings_field="endings", 
    label_field="label", 
    doc_token=4, 
    num_choices=4,
    prefix="Decide which of the following sentences makes the most sense: [SEP]"
):
    if prefix is None:
        prefix = ""

    def proc(examples):
        def fmt_labels(def_val=0., lab_val=1.):
            ll = []
            for n in examples[label_field]:
                c = [def_val] * num_choices
                c[int(n)] = lab_val
                ll.append(c)
            return ll

        ctx = examples[ctx_field]
        eds = examples[endings_field]
        labels = examples[label_field]
        ips = []
        dec_ips = []
        ll = []
        for c, e, l in zip(ctx, eds, labels):
            ll.append(int(l))
            cstr = prefix 
            for i, n in enumerate(e):
                #cstr += "[SEP]" + c + " " + n
                cstr += c + " " + n + "[SEP]"
                #cstr += " " + c + " " + n
                if i == int(l):
                    dec_ips.append(c + " " + n)
            #cstr += "[DOC]"
            ips.append(cstr)
        encoding = tokenizer(ips, padding="max_length", truncation=True, max_length=max_length)
        
        #encoding["labels"] = fmt_labels()
        encoding["labels"] = one_hot(ll, num_choices)

        dec_ii = tokenizer(dec_ips, padding="max_length", truncation=True, max_length=max_length)["input_ids"]
        encoding["decoder_input_ids"] = dec_ii
        #encoding["decoder_input_ids"] = encoding["input_ids"]
        #teacher_forcing_prefix = ""
        #encoding["decoder_input_ids"] = tokenizer([teacher_forcing_prefix + str(n) for n in encoding["labels"]], padding="max_length", truncation=True, max_length=max_length)["input_ids"]

        return encoding

    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_multiple_choice(
    dataset, 
    tokenizer,
    max_length, 
    num_proc=4, 
    remove_columns=None, 
    ctx_field="ctx", 
    endings_field="endings", 
    label_field="label", 
    doc_token=4, 
    num_choices=4,
    prefix=None
):
    if prefix is None:
        prefix = ""

    def proc(examples):
        def fmt_labels(def_val=0., lab_val=1.):
            ll = []
            for n in examples[label_field]:
                c = [def_val] * num_choices
                c[int(n)] = lab_val
                ll.append(c)
            return ll

        ctx = examples[ctx_field]
        eds = examples[endings_field]
        ips = [[prefix + c + " " + n + "[DOC]" for n in e] for c, e in zip(ctx, eds)]
        #encoding = [tokenizer(n, padding="max_length", truncation=True, max_length=max_length) for n in ips]
        encoding = {}
        for n in ips:
            cenc = tokenizer(n, padding="max_length", truncation=True, max_length=max_length)
            for k, v in cenc.items():
                if k not in encoding:
                    encoding[k] = []
                encoding[k].append(v)
        
        #encoding["labels"] = [int(n) for n in examples[label_field]]

        #encoding["decoder_input_ids"] = encoding["input_ids"].copy()
        #ll = [tokenizer([l] * num_choices, padding="max_length", truncation=True, max_length=max_length)["input_ids"] for l in examples[label_field]]
        ll = [ii[int(l)] for ii, l in zip (encoding["input_ids"], examples[label_field])]
        encoding["decoder_input_ids"] = ll

        #encoding["labels"] = ll
        encoding["labels"] = fmt_labels()

        return encoding
    
    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_maskedlm(
    dataset, 
    tokenizer, 
    max_length, 
    num_proc=4, 
    remove_columns=None, 
    to_mask=.15, 
    text_field="text", 
    mask_only_decoder_ids=False, 
    chance_rand_token=.2, 
    group_texts=True, 
    doc_token=4, 
    udoc_token=5, 
    mask_token=None,
    pad_token=None, 
    sparsify=False, 
    prefix=None,
    switch_ii_decoder_ii=False,
):
    vocab_size = tokenizer.vocab_size
    if mask_token is None:
        mask_token = tokenizer.mask_token_id
    if pad_token is None:
        pad_token = tokenizer.pad_token_id

    def proc(examples):
        text = examples[text_field]
        if group_texts:
            encoding = tokenizer(text)
            #encoding = tokenizer([" ".join(x) for x in text])
        else:
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length) # try max_length=512
        
        if prefix is not None:
            if group_texts:
                prefix_tokens = tokenizer(prefix)
            else:
                raise NotImplementedError()
        else:
            prefix_tokens = None

        if group_texts:
            encoding = _group_examples(encoding, max_length, sparsify, pad_token=pad_token, prefix_tokens=prefix_tokens)

        if mask_only_decoder_ids:
            raise NotImplementedError()
            encoding["input_ids"], encoding["labels"] = mask_tokens(
                input_ids=encoding["input_ids"], 
                vocab_size=vocab_size, 
                to_mask=to_mask, 
                mask_token=mask_token, 
                return_mask_only_decoder_ids=True, 
                chance_rand_token=chance_rand_token,
                ignore_up_to=mask_token
            )
        else:
            inp_ids = encoding["input_ids"].copy()
            encoding["labels"] = encoding["input_ids"].copy()
            #encoding["decoder_input_ids"] = []
            #encoding["decoder_input_ids"].append(encoding["input_ids"].copy())
            
            tok_mask = mask_tokens(
                input_ids=inp_ids.copy(), # don't know if mask_tokens() mutates input_ids
                vocab_size=vocab_size, 
                to_mask=to_mask, 
                mask_token=mask_token, 
                return_mask_only_decoder_ids=False, 
                chance_rand_token=chance_rand_token,
                ignore_up_to=mask_token
            )
            if switch_ii_decoder_ii:
                encoding["decoder_input_ids"] = tok_mask
                encoding["input_ids"] = inp_ids.copy()
            else:
                encoding["decoder_input_ids"] = inp_ids.copy() # don't ask
                encoding["input_ids"] = tok_mask
        return encoding

    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_translation(
    dataset, 
    tokenizer, 
    max_length, 
    source_lang, 
    target_lang, 
    prefix=None, 
    num_proc=4, 
    remove_columns=None,
    doc_token=4,
    udoc_token=5,
    pad_token=None,
    switch_ii_decoder_ii=False,
    num_sparse_token=1,
):
    if pad_token is None:
        pad_token = tokenizer.pad_token_id

    if prefix is not None:
        prefix_tokens = tokenizer(prefix)
    else:
        prefix_tokens = None

    def proc(examples):
        src = examples[source_lang]
        encoding = tokenizer(src)#, padding="max_length", truncation=True, max_length=max_length)
        tgt = examples[target_lang]
        tgt_enc = tokenizer(tgt)#, padding="max_length", truncation=True, max_length=max_length)["input_ids"]
        encoding["labels"] = tgt_enc["input_ids"]
        encoding = _zip_examples(encoding, max_length, "input_ids", "labels", True, pad_token, doc_token, udoc_token, prefix_tokens, num_sparse_token)
        if switch_ii_decoder_ii:
            encoding["decoder_input_ids"] = encoding["input_ids"].copy()
            encoding["input_ids"] = encoding["labels"].copy()
        else:
            encoding["decoder_input_ids"] = encoding["labels"].copy()
        #encoding["decoder_input_ids"] = encoding["input_ids"].copy()
        #encoding = _group_examples(encoding, max_length, sparsify=True)#, prefix_tokens=prefix_tokens)
        #encoding = _zip_examples(encoding, max_length, "input_ids", "labels", True)
        #encoding["decoder_input_ids"] = encoding["input_ids"].copy()
        return encoding
    
    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    # shift inputs to the right
    if is_torch_fx_proxy(input_ids):
        # Item assignment is not supported natively for proxies.
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
    else:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    # for hf models
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def preprocess_for_causallm(dataset, tokenizer, block_size=128, num_proc=4, remove_columns=None, text_field="text", 
    shift_right=True, decoder_start_token_id=0, pad_token_id=0
):
    def proc(examples):
        return tokenizer(examples[text_field])
        #return tokenizer([" ".join(x) for x in examples[text_field]])

    def group_texts(examples):
        examples = tokenizer(examples[text_field])

        result = _group_examples(examples, block_size)

        result["decoder_input_ids"] = result["input_ids"].copy()
        if shift_right:
            result["labels"] = _shift_right(result["input_ids"].clone(), decoder_start_token_id, pad_token_id)
        else:
            result["labels"] = result["input_ids"].copy()#.clone()
        return result

    #tok_ds = dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)
    #return tok_ds.map(group_texts, batched=True, num_proc=num_proc)
    return dataset.map(group_texts, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def preprocess_for_monologe(dataset, tokenizer, max_length, num_proc=4, text_field="text", remove_columns=None):
    def proc(examples):
        text = examples[text_field]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        encoding["labels"] = encoding["input_ids"].copy()
        encoding["decoder_input_ids"] = encoding["input_ids"].copy()
        return encoding
    
    return dataset.map(proc, batched=True, num_proc=num_proc, remove_columns=remove_columns)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def pos_diff(a, b):
    return [(x - y) for x, y in zip(a, b)]


def max_diff(l):
    return max(max([[x - y for y in l] for x in l]))
    

class BatchBuffer:
    def __init__(self, dataset, batch_size):
        self.ds = dataset
        self.bs = batch_size
        if isinstance(dataset, Dataset):
            self.schema = list(dataset.features.keys())
        if isinstance(dataset, dict):
            self.schema = list(dataset.keys())

    def _b_to_ten(self, b):
        ret = []
        for n in b.values():
            ret.append(torch.tensor(n))
        return ret

    def shuffle(self):
        self.ds = self.ds.shuffle()
        return self

    def __len__(self):
        return int(len(self.ds) / self.bs)

    def __iter__(self):
        batch = {}
        for step, n in enumerate(self.ds):
            for k, v in n.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)
            if (step + 1) % self.bs != 0:
                continue
            yield self._b_to_ten(batch)
            batch = {}


# only a short term solution to avoid breaking legacy code
STRING_BATCH_INDEX = False


def _get_batch_item(batch, batch_schema, item, device):
    if item not in batch_schema:
        return None
    if STRING_BATCH_INDEX:
        idx = item
    else:
        idx = batch_schema.index(item)
    return batch[idx].to(device)
    

def _get_model_args(batch, batch_schema, items, device):
    ret = {}
    for n in items:
        e = _get_batch_item(batch, batch_schema, n, device)
        if e is not None:
            ret[n] = e
    return ret


def mask_tokens(input_ids, vocab_size, to_mask=0.15, mask_token=4, return_mask_only_decoder_ids=False, chance_rand_token=0.2, ignore_up_to=4):
    """
    https://www.analyticsvidhya.com/blog/2022/09/fine-tuning-bert-with-masked-language-modeling/
    """
    inp_ids = []
    if return_mask_only_decoder_ids:
        dec_inp_ids = []
    lbs = []
    idx = 0
    for inp in np.array(input_ids):#.numpy():
        #actual_tokens = list(set(range(100)) - 
        #                    set(np.where((inp == 101) | (inp == 102) 
        #                        | (inp == 0))[0].tolist()))

        actual_tokens = list(set(range(len(inp))) - 
                            set(np.where((inp <= ignore_up_to))[0].tolist()))
                            #set(np.where((inp == 0))[0].tolist()))
        #We need to select 15% random tokens from the given list
        num_of_token_to_mask = int(len(actual_tokens) * to_mask)
        token_to_mask = np.random.choice(np.array(actual_tokens), 
                                        size=num_of_token_to_mask, 
                                        replace=False).tolist()
        #Now we have the indices where we need to mask the tokens
        if return_mask_only_decoder_ids:
            dec_inp = np.array([mask_token] * len(inp))
            dec_inp[token_to_mask] = inp[token_to_mask]
            dec_inp_ids.append(dec_inp)
        if random.random() < chance_rand_token:
            inp[token_to_mask] = random.randint(ignore_up_to+1, vocab_size)
        else:
            inp[token_to_mask] = mask_token
        inp_ids.append(inp)
        idx += 1
    if return_mask_only_decoder_ids:
        return inp_ids, dec_inp_ids
    return inp_ids


def fill_masks(
    model, 
    dataloader,
    batch_schema,
    device,
    mask_token_id=4,
    replace_input_ids=False,
    is_hf_model=False,
    is_encoder_decoder_model=False
):
    tgt_input_ids = []
    src_input_ids = []
    src_token_type_ids = []
    src_attention_mask = []
    src_labels = []
    model.eval()
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = _get_batch_item(batch, batch_schema, "input_ids", device)
        token_type_ids = _get_batch_item(batch, batch_schema, "token_type_ids", device)
        attention_mask = _get_batch_item(batch, batch_schema, "attention_mask", device)
        labels = _get_batch_item(batch, batch_schema, "labels", device)
        with torch.no_grad():
            if is_encoder_decoder_model:
                raise NotImplementedError("filling for encoder-decoder models is not implemented yet")
            else:
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        logits = outputs.logits if is_hf_model else outputs
        mask_token_index = torch.where(input_ids == mask_token_id)[1]
        for i, l in enumerate(logits):
            mask_token_logits = l[mask_token_index, :]
            tok = torch.topk(mask_token_logits, 1, dim=1)
            tok = tok.indices.tolist()
            midx = torch.where(input_ids[i] == mask_token_id)[0].tolist()
            rep_ids = input_ids[i].tolist()
            for j, m in zip(midx, tok):
                rep_ids[j] = m[0]
            tgt_input_ids.append(rep_ids)
            src_input_ids.append(input_ids[i].tolist())
            if token_type_ids is not None:
                src_token_type_ids.append(token_type_ids[i].tolist())
            if attention_mask is not None:
                src_attention_mask.append(attention_mask[i].tolist())
            if labels is not None:
                src_labels.append(labels[i].tolist())
    if replace_input_ids:
        ret = {
            "input_ids": tgt_input_ids,
        }
    else:
        ret = {
            "input_ids": src_input_ids,
            "filled_input_ids": tgt_input_ids
        }
    if len(src_token_type_ids) > 0:
        ret["token_type_ids"] = src_token_type_ids
    if len(src_attention_mask) > 0:
        ret["attention_mask"] = src_attention_mask
    if len(src_labels) > 0:
        ret["labels"] = src_labels
    return ret


def dump_model_to_file(
    path, 
    model, 
    train_stats, 
    test_stats, 
    epochs, 
    done_epochs, 
    id2label,
    is_hf_model=False,
    dump_model=True,
    encoder_decoder_model=False,
    dump_coin_regions=False,
    coin_region_lambda=None,
    safe_serialization=False,
):
    try:
        os.mkdir(path)
    except OSError as err:
        print(err)
    if encoder_decoder_model:
        encoder_path = f"{path}/encoder/"
        decoder_path = f"{path}/decoder/"
        try:
            os.mkdir(encoder_path)
        except OSError as err:
            print(err)
        try:
            os.mkdir(decoder_path)
        except OSError as err:
            print(err)
    if dump_coin_regions:
        assert coin_region_lambda is not None
        try:
            os.mkdir(f"{path}/regions/")
        except OSError as err:
            print(err)
        for i, r in enumerate(coin_region_lambda(model)):
            r.save_pretrained(f"{path}/regions/region_{i}", safe_serialization=safe_serialization)
    if dump_model:
        #if is_hf_model:
        model.save_pretrained(f"{path}/model", safe_serialization=safe_serialization)
        if encoder_decoder_model:
            model.encoder.save_pretrained(encoder_path, safe_serialization=safe_serialization)
            model.decoder.save_pretrained(decoder_path, safe_serialization=safe_serialization)
        #else:
        #    torch.save(model.state_dict(), f"{path}/model")
        #torch.save(model.bern.s_n_ls, f"{path}/s_n_ls")
        #torch.save(model.bern.c_n_ls, f"{path}/c_n_ls")
    mdict = {
        "train_stats": [n.__dict__ for n in train_stats],
        "test_stats": [n.__dict__ for n in test_stats],
        "epochs": epochs,
        "done_epochs": done_epochs,
        "id2label": id2label
    }
    with open(f"{path}/report.json", "w") as f:
        json.dump(mdict, f)
    print("model dumped")


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def emb2idx(X, emb):
    res = []
    #print(emb.weight.data.shape, X.shape)
    for i in range(X.shape[1]):
        #print(emb.weight.data.shape, n.shape, n[0].shape, X[:, 0:1, :].shape)
        distance = torch.norm(emb.weight.data - X[:, i:(i+1), :], dim=1)
        #print(distance)
        nearest = torch.argmin(distance, dim=-1)
        #print(nearest)
        res.append(nearest)
    ret = torch.stack(res, dim=1)
    #print(ret.shape)
    return ret

    
def decode_mlm_logits(logits, tokenizer, input_ids, emb, use_argmax=False):
    #if use_argmax:
    return tokenizer.batch_decode(torch.argmax(logits, dim=-1))
    k = 1
    return tokenizer.batch_decode(logits.topk(k, -1).indices[..., -1])
    #return tokenizer.batch_decode(emb2idx(logits, emb))

    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    ii_ten = input_ids
    #logits = logits.logits
    res = []
    res_dec = []
    for i, l in enumerate(logits):
        mask_token_logits = l[mask_token_index, :]
        tok = torch.topk(mask_token_logits, 1, dim=1)
        tok = tok.indices.tolist()
        midx = torch.where(input_ids[i] == tokenizer.mask_token_id)[0].tolist()
        rep_ids = input_ids[i].tolist()
        for i, m in zip(midx, tok):
            rep_ids[i] = m[0]
        res.append(rep_ids)
        #res_dec.append(tokenizer.decode(rep_ids))
    res_dec = [tokenizer.decode(r) for r in res]
    return res_dec


def _detach_hidden_states(S):
    if S is None:
        return S
    if isinstance(S, torch.Tensor):
        return S.detach()
    if isinstance(S, list):
        return [_detach_hidden_states(n) for n in S]
    raise ValueError(f"S is neither a list nor a Tensor but {S.__class__}")


def train_set_epoch(
    model: nn.Module,
    optimizer,
    scheduler,
    train_dataloader: Union[DataLoader, BatchBuffer],
    batch_schema: List[str],
    device,
    loss_function,
    id2label,
    masked_lm_task=False,
    vocab_size=None,
    print_status=True,
    is_hf_model=False,
    batch_size=-1,
    one_label_only=True,
    mixed_lm_task=False,
    mixed_lm_loss_function=None,
    is_encoder_decoder_model=False,
    calc_metrics=True,
    electra_task=False,
    empty_cache=False,
    retain_graph=False,
    batch_hack=False,
    imitation_model=None,
    causal_lm=False,
    generic_output_class=False,
    forward_args=["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "labels"],
    mlm_decode_n=0,
    tokenizer=None,
    mlm_decode_max_chars=100,
    mlm_decode_max_batch_output=2,
    check_run=False,
    chomsky_task=False,
):
    if mixed_lm_task or causal_lm:
        masked_lm_task = True

    if masked_lm_task and vocab_size is None:
        raise ValueError("vocab_size has to be specified when training for a masked lm task")

    c_train_stats = EpochStats("train", id2label, chomsky_task)
    t0 = time.time()
    if check_run:
        print("CHECK RUN")
        model.eval()
    else:
        model.train()

    #print_n = (len(train_dataloader) / 50) if len(train_dataloader) > 10000 else (int(len(train_dataloader) / 40) if len(train_dataloader) > 400 else 20)
    #if chomsky_task:
    #    len_dl = len(train_dataloader) // batch_size 
    #else:
    len_dl = len(train_dataloader)
    print_n = len_dl // 50 #if len_dl > 500 else 50

    mems = None
    S, C = None, None
    encoder_hidden_state = None

    last_elapse = 0
    for step, batch in enumerate(train_dataloader):
        #print(step, print_n, step % print_n)
        if print_status and (step % print_n == 0) and not step == 0:
            c_time = time.time()
            elapsed = c_time - t0
            last_step_time = int(round(elapsed - last_elapse))
            remaining = (len(train_dataloader) - step) / print_n * last_step_time
            last_elapse = elapsed
            print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:>8}, Remaining: {:>8}.".format(step, 
                                                                                            len(train_dataloader), 
                                                                                            format_time(elapsed), 
                                                                                            format_time(remaining)))

        #print(len(batch), batch.index("masked_input_ids"))

        #input_ids = _get_batch_item(batch, batch_schema, "input_ids", device)
        #token_type_ids = _get_batch_item(batch, batch_schema, "token_type_ids", device)
        #attention_mask = _get_batch_item(batch, batch_schema, "attention_mask", device)
        model_args = _get_model_args(batch, batch_schema, forward_args, device)
        #print(forward_args)
        #print(model_args.keys())

        if imitation_model is not None:
            with torch.no_grad():
                labels = imitation_model(**model_args)
                if is_hf_model:
                    labels = labels.logits
        else:
            labels = _get_batch_item(batch, batch_schema, ("masked_input_ids" if mixed_lm_task else "labels"), device)
        
        # HACK: workaround if the model doesn't account for different batch sizes
        B = model_args["input_ids"].shape[0]
        if batch_hack and batch_size > 0 and B != batch_size:
            continue
        #if batch_hack and batch_size > 0 and labels.shape != (batch_size, len(id2label)):
        #    continue

        model.zero_grad()
        if is_hf_model:
            outputs = model(**model_args)
            logits = outputs.logits
            loss = outputs.loss
            aux_loss = None
        elif generic_output_class:
            outputs = model(
                S=S,
                C=C,
                encoder_hidden_state=encoder_hidden_state,
                **model_args
            )
            logits = outputs.logits
            encoder_hidden_state = outputs.encoder_hidden_state
            loss = outputs.loss
            aux_loss = outputs.aux_loss
            S = outputs.S
            C = outputs.C
            if encoder_hidden_state is not None:
                encoder_hidden_state = _detach_hidden_states(encoder_hidden_state)
            if S is not None:
                S = _detach_hidden_states(S)
            if C is not None:
                C = _detach_hidden_states(C)
        else:
            logits, S, C, decoder_logits, aux_loss = model(
                S=S,
                C=C,
                #mems=mems, 
                **model_args)
            if S is not None:
                S = _detach_hidden_states(S)
            if C is not None:
                C = _detach_hidden_states(C)
            loss = None
            aux_loss = None

        dec_mark = random.random()
        dec_log = dec_mark <= mlm_decode_n
        #print(dec_mark, mlm_decode_n, dec_log)
        #if mlm_decode_n > 0 and (step % mlm_decode_n == 0):
        if mlm_decode_n > 0 and dec_log:
            if tokenizer is None:
                raise ValueError("A tokenizer is needed to decode sample logits")
            
            #emb = model.rrb.decoder_embeddings[0].word_embeddings

            print("\nInternal decoder outputs:")
            for i, dl in enumerate(decoder_logits):
                #if model.rrb.config.decoder_pipeline:
                dei = 0
                #else:
                #    dei = i
                #emb = model.rrb.decoder_embeddings[dei].word_embeddings
                emb = model.bounce.embeddings.word_embeddings
                #decode_logits = decode_mlm_logits(logits, tokenizer, model_args["input_ids"])
                decode_logits = decode_mlm_logits(dl, tokenizer, None, emb)
                decode_logits = [n[:mlm_decode_max_chars] for n in decode_logits[:mlm_decode_max_batch_output]]
                print(f"  Decoder {i}: {decode_logits}")
            decode_output_logits = decode_mlm_logits(logits, tokenizer, None, emb, True)
            decode_output_logits = [n[:mlm_decode_max_chars] for n in decode_output_logits[:mlm_decode_max_batch_output]]
            print(f"Model decoder output: {decode_output_logits}")
            ref_by = tokenizer.batch_decode(model_args["input_ids"])
            ref_by = [n[:mlm_decode_max_chars] for n in ref_by[:mlm_decode_max_batch_output]]
            print(f"Model input: {ref_by}\n")

        #print(logits.shape, labels.shape)
        #print(logits[..., -1].shape)
        #for n in logits:
        #    print(n.shape)
        #print(labels[0])
        #print(labels.view(-1)[0])
        #print(logits)
        #print(torch.min(logits), torch.max(logits))
        if loss is None:
            if masked_lm_task:
                if causal_lm:
                    logits = logits[:, :-1, :].contiguous()
                    labels = labels[:, 1:].contiguous()

                if mixed_lm_task:
                    loss = mixed_lm_loss_function(logits.view(-1, vocab_size), labels.view(-1))
                else:
                    #print(logits.shape, logits.view(-1).shape)
                    #print(labels.shape, labels.view(-1).shape)
                    #print(labels)
                    loss = loss_function(logits.view(-1, vocab_size), labels.view(-1))
                    #for i, dl in enumerate(decoder_logits):
                    #    loss += loss_function(dl.view(-1, vocab_size), labels.view(-1))
                    #loss = loss_function(logits.view(-1), labels.view(-1))
            elif electra_task:
                if "attention_mask" in model_args:
                    attention_mask = model_args["attention_mask"]
                    active_loss = attention_mask.view(-1, logits.shape[1]) == 1
                    active_logits = logits.view(-1, logits.shape[1])[active_loss]
                    active_labels = labels[active_loss]
                    loss = loss_function(active_logits, active_labels.float())
                else:
                    loss = loss_function(logits.view(-1, logits.shape[1]), labels.float())
            else:
                #print(logits.shape, labels.shape)
                loss = loss_function(logits, labels)

            if aux_loss is not None:
                loss += aux_loss

        if chomsky_task:
            logits = logits.argmax(-1)
        #logits = F.one_hot(logits, vocab_size)
        #print(logits.shape, labels.shape)
        #print(logits.dtype, labels.dtype)
        #print(logits[0])
        #print(labels[0])
        #acc_2 = (logits == labels).float().mean().item()
        #c_train_stats.add_score("accuracy_2", acc_2)

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        c_train_stats.add_score("loss", loss.item())
        if not masked_lm_task and not electra_task and calc_metrics:
            c_train_stats.flat_metrics(logits, labels, one_label_only=one_label_only)
        loss.backward(retain_graph=retain_graph)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        if empty_cache:
            torch.cuda.empty_cache()
        
    if print_status:
        print("")
        print("  Average training scores:")
        for k, v in c_train_stats.calc_avg_scores().items():
            print(f"    {k}: {v}")
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    return c_train_stats


def test_set_epoch(
    model: nn.Module,
    test_dataloader: Union[DataLoader, BatchBuffer],
    batch_schema: List[str],
    device,
    loss_function,
    id2label,
    masked_lm_task=False,
    vocab_size=None,
    print_status=True,
    is_hf_model=False,
    batch_size=-1,
    one_label_only=True,
    mixed_lm_task=False,
    mixed_lm_loss_function=None,
    is_encoder_decoder_model=False,
    calc_metrics=True,
    electra_task=False,
    empty_cache=False,
    batch_hack=False,
    causal_lm=False,
    generic_output_class=False,
    forward_args=["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "labels"],
    chomsky_task=False,
):
    if mixed_lm_task or causal_lm:
        masked_lm_task = True
    
    if masked_lm_task and vocab_size is None:
        raise ValueError("vocab_size has to be specified when testing for a masked lm task")

    c_test_stats = EpochStats("test", id2label, chomsky_task)
    t0 = time.time()
    model.eval()

    S, C = None, None
    encoder_hidden_state = None

    for batch in test_dataloader:
        #input_ids = _get_batch_item(batch, batch_schema, "input_ids", device)
        #token_type_ids = _get_batch_item(batch, batch_schema, "token_type_ids", device)
        #attention_mask = _get_batch_item(batch, batch_schema, "attention_mask", device)
        labels = _get_batch_item(batch, batch_schema, ("masked_input_ids" if mixed_lm_task else "labels"), device)
        model_args = _get_model_args(batch, batch_schema, forward_args, device)

        # HACK: workaround if the model doesn't account for different batch sizes
        B = model_args["input_ids"].shape[0]
        if batch_hack and batch_size > 0 and B != batch_size:
            continue
        #if batch_hack and batch_size > 0 and labels.shape != (batch_size, len(id2label)):
        #    continue

        if (not masked_lm_task) and batch_size > 0 and model_args["input_ids"].shape[0] != batch_size:
            continue

        with torch.no_grad():
            if is_hf_model:
                outputs = model(**model_args)
                logits = outputs.logits
                loss = outputs.loss
                aux_loss = None
            elif generic_output_class:
                outputs = model(
                    S=S,
                    C=C,
                    encoder_hidden_state=encoder_hidden_state,
                    **model_args
                )
                logits = outputs.logits
                encoder_hidden_state = outputs.encoder_hidden_state
                loss = outputs.loss
                aux_loss = outputs.aux_loss
                S = outputs.S
                C = outputs.C
                if S is not None:
                    S = _detach_hidden_states(S)
                if C is not None:
                    C = _detach_hidden_states(C)
            else:
                logits, S, C, decoder_logits, aux_loss = model(
                    S=S,
                    C=C,
                    #mems=mems, 
                    **model_args)
                if S is not None:
                    S = _detach_hidden_states(S)
                if C is not None:
                    C = _detach_hidden_states(C)

        if loss is None:
            if masked_lm_task:
                if causal_lm:
                    logits = logits[:, :-1, :].contiguous()
                    labels = labels[:, 1:].contiguous()

                if mixed_lm_task:
                    loss = mixed_lm_loss_function(logits.view(-1, vocab_size), labels.view(-1))
                else:
                    loss = loss_function(logits.view(-1, vocab_size), labels.view(-1))
            elif electra_task:
                if "attention_mask" in model_args:
                    attention_mask = model_args["attention_mask"]
                    active_loss = attention_mask.view(-1, logits.shape[1]) == 1
                    active_logits = logits.view(-1, logits.shape[1])[active_loss]
                    active_labels = labels[active_loss]
                    loss = loss_function(active_logits, active_labels.float())
                else:
                    loss = loss_function(logits.view(-1, logits.shape[1]), labels.float())
            else:
                loss = loss_function(logits, labels)
            
            if aux_loss is not None:
                loss += aux_loss

        if chomsky_task:
            logits = logits.argmax(-1)

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        c_test_stats.add_score("loss", loss.item())
        if not masked_lm_task and not electra_task and calc_metrics:
            c_test_stats.flat_metrics(logits, labels, one_label_only=one_label_only)

        if empty_cache:
            torch.cuda.empty_cache()

    if print_status:
        print("  Average testing scores:")
        for k, v in c_test_stats.calc_avg_scores().items():
            print(f"    {k}: {v}")
        print("  Testing took: {:}".format(format_time(time.time() - t0)))
    
    return c_test_stats


def train_bern_model(
    model: nn.Module,
    optimizer,
    scheduler,
    epochs,
    device,
    loss_function,
    id2label,
    batch_schema: Optional[List[str]]=None,
    train_dataloader: Optional[Union[DataLoader, BatchBuffer]]=None,
    test_dataloader: Optional[Union[DataLoader, BatchBuffer]]=None,
    create_train_dataloader: Optional[Callable]=None,
    create_test_dataloader: Optional[Callable]=None,
    masked_lm_task=False,
    vocab_size=None,
    print_status=True,
    is_hf_model=False,
    checkpoint_path: Optional[str]=None,
    train_batch_size=-1,
    test_batch_size=-1,
    only_save_core=False,
    one_label_only=True,
    mixed_lm_task=False,
    mixed_lm_loss_function=None,
    epoch_i=0,
    is_encoder_decoder_model=False,
    calc_metrics=True,
    electra_task=False,
    empty_cache=False,
    retain_graph=False,
    batch_hack_train=False,
    batch_hack_test=False,
    imitation_model=None,
    add_layers_on_stagnation=False,
    num_layers_to_add=1,
    add_layers_threshold=0.005,
    plot_k_topics=False,
    causal_lm=False,
    generic_output_class=False,
    forward_args=["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "labels"],
    mlm_decode_n=0,
    tokenizer=None,
    mlm_decode_max_chars=100,
    mlm_decode_max_batch_output=2,
    dump_coin_regions=False,
    coin_region_lambda=None,
    check_run=False,
    chomsky_task=False,
):
    assert train_dataloader is not None or create_train_dataloader is not None
    assert test_dataloader is not None or create_test_dataloader is not None
    train_stats, test_stats = [], []
    if plot_k_topics:
        kts_l = []
    #for epoch_i in range(epochs):
    while epoch_i < epochs:
        if print_status:
            print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        
        if create_train_dataloader is not None:
            if print_status:
                print("\nCreating train dataloader...")
            train_dataloader = None # please don't ask why
            train_dataloader = create_train_dataloader()

        if create_test_dataloader is not None:
            if print_status:
                print("\nCreating test dataloader...")
            test_dataloader = None # seriously, don't
            test_dataloader = create_test_dataloader()

        if batch_schema is None:
            batch_schema = train_dataloader.schema
            if print_status:
                print("\nGenerated batch schema as", batch_schema)

        if print_status:
            print("\nTraining...")

        if mixed_lm_task:
            mlm_use = (epoch_i % 2 == 0)
            model.set_lm_use(mlm_use)

        c_train_stats = train_set_epoch(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            batch_schema,
            device,
            loss_function,
            id2label,
            masked_lm_task,
            vocab_size,
            print_status,
            is_hf_model,
            train_batch_size,
            one_label_only,
            mlm_use if mixed_lm_task else False,
            mixed_lm_loss_function,
            is_encoder_decoder_model,
            calc_metrics,
            electra_task,
            empty_cache,
            retain_graph,
            batch_hack_train,
            imitation_model,
            causal_lm,
            generic_output_class,
            forward_args,
            mlm_decode_n,
            tokenizer,
            mlm_decode_max_chars,
            mlm_decode_max_batch_output,
            check_run,
            chomsky_task
        )
        train_stats.append(c_train_stats)

        if print_status:
            print("\nRunning Testing...")
        c_test_stats = test_set_epoch(
            model,
            test_dataloader,
            batch_schema,
            device,
            loss_function,
            id2label,
            masked_lm_task,
            vocab_size,
            print_status,
            is_hf_model,
            test_batch_size,
            one_label_only,
            mlm_use if mixed_lm_task else False,
            mixed_lm_loss_function,
            is_encoder_decoder_model,
            calc_metrics,
            electra_task,
            empty_cache,
            batch_hack_test,
            causal_lm,
            generic_output_class,
            forward_args,
            chomsky_task
        )
        test_stats.append(c_test_stats)
        
        if checkpoint_path is not None:
            dump_model_to_file(
                f"{checkpoint_path}/epoch_{epoch_i}/",
                #(model.bert if is_hf_model else model.bern) if only_save_core else model,
                model,
                train_stats,
                test_stats,
                epochs,
                epoch_i,
                id2label,
                is_hf_model,
                dump_model=True,
                encoder_decoder_model=is_encoder_decoder_model,
                dump_coin_regions=dump_coin_regions,
                coin_region_lambda=coin_region_lambda,
            )
        
        if plot_k_topics:
            kts = model.calc_topic_block_dist()
            plt.title(f"% call per topic block for epoch {epoch_i + 1}")
            plt.xlabel("num block")
            plt.ylabel("%")
            plt.ylim(0, 1)
            plt.plot(list(range(1, len(kts) + 1)), kts)
            plt.show()
            print("All values      :", kts)
            kts_l.append(kts)
            if len(kts_l) >= 2:
                print("Diff to previous:", pos_diff(kts_l[-1], kts_l[-2]))
            print("Max diff in current batch:", max_diff(kts))
            if not model.config.fixed_k:
                print("\nAvg called blocks: {:.2f} / {:}".format(*model.calc_avg_block_calls()))
        
        if add_layers_on_stagnation and len(test_stats) >= 2:
            loss_diff = test_stats[-2].calc_avg_scores()["loss"] - test_stats[-1].calc_avg_scores()["loss"]
            if loss_diff < add_layers_threshold:
                model.add_layers(device, num_layers_to_add)

        epoch_i += 1

    print("\nTraining complete!")
    return train_stats, test_stats

