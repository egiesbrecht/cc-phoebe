import glob
import pandas as pd
from datasets import Dataset, concatenate_datasets
import numpy as np
import torch
import torch.nn.functional as F
import itertools


def load_set(paths, use_one_label_only=False, iloc_limit: int=None, as_dataset=True, drop_unused=True, drop_labels=False, unused_fields=None):
    files = []
    for p in paths:
        files += glob.glob(p)
    print("loading files")
    for f in files:
        print(f"    {f}")
    li = [pd.read_csv(fname, index_col=None, header=0) for fname in files]
    frame = pd.concat(li, axis=0, ignore_index=True)
    #if not use_one_label_only and drop_unused:
    #    frame.drop(labels=["author", "subreddit", "style"], axis=1, inplace=True)
    if unused_fields is not None:
        frame.drop(labels=unused_fields, axis=1, inplace=True)
    if drop_labels:
        frame.drop(labels=[n for n in frame if n != "text"], axis=1, inplace=True)
    if iloc_limit:
        frame = frame.iloc[:iloc_limit, :]
    if as_dataset:
        return Dataset.from_pandas(frame, preserve_index=False)
    else:
        return frame


def load_moses_set(path_schema: dict[list[str]]):
    ds = {}
    for k, v in path_schema.items():
        cl = []
        cfiles = []
        for p in v:
            cfiles += glob.glob(p)
        for file in cfiles:
            with open(file) as f:
                cl += [l.rstrip() for l in f]
        ds[k] = cl
    return Dataset.from_dict(ds)


def generate_ch_batches(generate_set_fn, B, T_train, T_test, vocab_size, num_train_samples, num_test_samples, sample_method="linspace", num_warmup_steps=1000):
    """
    sample_method = "logspace" | "uniform" | "static" | "static-warmup"
    """
    #assert num_train_samples % B == 0, f"{num_train_samples} % {B} != 0"
    if sample_method == "uniform":
        #U = ((1 - (T_train / 2)) * torch.rand(num_train_samples // B) + (T_train / 2)).to(int) * 2 # ensure that (n in U) is divisible by 2
        U = torch.FloatTensor(num_train_samples // B).uniform_(1, T_train).int()
    elif sample_method == "linspace":
        U = torch.linspace(1, T_train, num_train_samples // B, dtype=int) 
    elif sample_method == "static-warmup":
        U = torch.cat((
            torch.linspace(1, T_train, num_warmup_steps // B, dtype=int),
            torch.full((((num_train_samples - num_warmup_steps) // B),), T_train)
        ))
    elif sample_method == "static":
        U = [T_train] * num_train_samples
    else:
        raise ValueError(sample_method)
    #U += U % 2
    print("sample schema:", U)
    train_buf = [generate_set_fn(B, n, vocab_size) for n in U]
    test_buf = [generate_set_fn(B, T_test, vocab_size) for _ in range(num_test_samples)]
    return train_buf, test_buf


def generate_bucket_sort_set(B: int, T: int, vocab_size: int):
    string = torch.randint(0, vocab_size, (B, T))
    sorted_string = string.sort().values
    #string = F.one_hot(string, vocab_size)
    #sorted_string = F.one_hot(sorted_string, vocab_size)
    return {
        "input_ids": string,
        "decoder_input_ids": sorted_string,
        "labels": sorted_string
    }


def generate_duplicate_string_set(B: int, T: int, vocab_size: int):
    assert T % 2 == 0, T
    cstring = torch.randint(0, vocab_size, (B, T//2))
    zeros = torch.zeros(B, T//2, dtype=torch.int64)
    string = torch.cat((cstring, zeros), 1)
    output = torch.cat((cstring, cstring), 1)
    return {
        "input_ids": string,
        "decoder_input_ids": output,
        "labels": output
    }


def generate_parity_check_set(B: int, T: int, vocab_size):
    string = torch.randint(0, 2, (B, T))
    #output = (string.sum(1) % 2 == 0).long()
    output = string.sum(1) % 2
    return {
        "input_ids": string,
        "decoder_input_ids": output.unsqueeze(-1),
        "attention_mask": string, 
        "labels": output
    }


def generate_missing_duplicate_string_set(B: int, T: int, vocab_size):
    if T == 1:
        dupl_string = torch.ones(B, T)
        dii = dupl_string.clone()
        output = torch.ones(B)
    else:
        #assert T % 2 == 0, T
        string = torch.randint(0, 2, (B, T//2))
        dupl_string = torch.cat((string, string), 1)
        indices = torch.randint(0, dupl_string.shape[1], (B,))
        output = []
        for i in range(B):
            c = dupl_string[i, indices[i]]
            output.append(c)
        output = torch.stack(output)
        dii = dupl_string.clone()
        #dii = string.clone()
        for i in range(B):
            dupl_string[i, indices[i]] = 2
    return {
        "input_ids": dupl_string,
        "decoder_input_ids": dii,
        #"attention_mask": None,#max(dupl_string, 1),
        "labels": output#F.one_hot(output, 2)
    }


def generate_static_parity_check_set(B, T_train, T_test, num_train_samples, num_test_samples):
    train_ii = []
    ls = torch.linspace(1, T_train, num_train_samples, dtype=int)
    #for i in range(1, num_samples+1):
    for i in ls:
        train_ii.append(torch.randint(0, 2, (B, i)))
    test_ii = [torch.randint(0, 2, (B, T_test)) for _ in range(num_test_samples)]
    train_labels = [x.sum(1) % 2 for x in train_ii]
    test_labels = [x.sum(1) % 2 for x in test_ii]
    train_o = [
        {
            "input_ids": ii,
            "labels": ll
        } for ii, ll in zip(train_ii, train_labels)
    ]
    test_o = [
        {
            "input_ids": ii,
            "labels": ll
        } for ii, ll in zip(test_ii, test_labels)
    ]
    return train_o, test_o

