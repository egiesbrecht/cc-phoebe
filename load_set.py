import glob
import pandas as pd
from datasets import Dataset, concatenate_datasets
import numpy as np
import torch
import torch.nn.functional as F
import itertools
import random
import functools
import math
import tree
import tqdm
import collections
from typing import List, Union
from model_training import segment_prod, segment_sum


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
        #U = torch.full((num_train_samples,), T_train)
    else:
        raise ValueError(sample_method)
    #print(U)
    #U += U % 2 #+ U % 4
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
    #assert T % 2 == 0, T
    if T == 1:
        string = torch.ones(B, T)
        output = torch.ones(B, T)
    else:
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


def generate_missing_duplicate_string_set(B: int, T: int, vocab_size: None):
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


def expression_from_numbers(numbers_n, numbers_m):
    """placeholder insertion"""
    return [n + [2] + m for n, m in zip(numbers_n, numbers_m)]
    #return [n + m for n, m in zip(numbers_n, numbers_m)]


def numbers_to_variable_length_binary(numbers: List[int], Ts: List[int], litte_endian=True):
    bin_strs = [f"{num:b}".zfill(t) for num, t in zip(numbers, Ts)]
    if litte_endian:
        bin_strs = [bin[::-1] for bin in bin_strs]
    return [list(map(int, bin)) for bin in bin_strs]


def numbers_to_fixed_length_binary(numbers: List[int], T: int, litte_endian=True):
    return numbers_to_variable_length_binary(numbers, [T] * len(numbers), litte_endian)


def _sample_expressions_and_results(B: int, T: int):
    if T <= 2:
        numbers = torch.randint(0, 2**T - 1, (B,))
        expressions = numbers_to_fixed_length_binary(numbers, T)
        results = numbers_to_fixed_length_binary(numbers, 0)
        return expressions, results
    
    length_n = torch.randint(1, T-1, (B,))
    length_m = T - 1 - length_n
    #length_n = length_m = torch.full((B, ), T // 2)
    #t4 = T // 4
    #tr = T - 1 - t4
    #length_m = torch.full((B,), t4)
    #length_n = torch.full((B,), tr)

    int_n = [random.randint(1, 2**int(len_n) - 1) for len_n in length_n]
    int_m = [random.randint(1, 2**int(len_m) - 1) for len_m in length_m]
    bin_n = numbers_to_variable_length_binary(int_n, length_n)
    bin_m = numbers_to_variable_length_binary(int_m, length_m)
    expressions = expression_from_numbers(bin_n, bin_m)
    int_sum = list(map(sum, zip(int_n, int_m)))
    results = numbers_to_fixed_length_binary(int_sum, 0)
    return expressions, results


def generate_binary_addition_set(B: int, T: int, vocab_size: None):
    # litte endian
    expressions, results = _sample_expressions_and_results(B, T+1)
    # padding
    #results = [res + [0] * (T - len(res)) for res in results]
    results = [res + [2] + [0] * (T - len(res)) for res in results]
    #results = [[0] * (T - len(res) + 1) + res for res in results]
    expressions = torch.Tensor(expressions)#, dtype=torch.int32)
    results = torch.Tensor(results)#, dtype=torch.int32)
    return {
        "input_ids": expressions,
        "labels": results
    }


def binary_addition_mask(target):
    B, T = target.shape[:2]
    term_idcs = np.argmax(
        np.argmax(target, axis=-1), axis=-1, keepdims=True
    )
    idcs = np.tile(np.arange(T), (B, 1))
    mask = idcs <= term_idcs
    return mask
    

def generate_binary_sqrt_set(B: int, T: int, vocab_size: None):
    # big endian
    numbers = [random.randint(1, 2**T - 1) for _ in range(B)]
    binary_numbers = numbers_to_fixed_length_binary(numbers, T, litte_endian=False)
    sqrts = list(map(math.isqrt, numbers))
    out_len = math.ceil(T / 2)
    binary_sqrts = torch.Tensor(numbers_to_fixed_length_binary(sqrts, out_len, litte_endian=False))
    binary_sqrts = torch.cat((torch.zeros(B, math.floor(T / 2)), binary_sqrts), 1)
    return {
        "input_ids": torch.Tensor(binary_numbers),
        "labels": (binary_sqrts)
    }


def binary_sqrt_mask(target):
    B, T = target.shape[:2]
    mask = np.concatenate((
        np.full((B, math.floor(T / 2)), 0),
        np.full((B, math.ceil(T / 2)), 1)
    ), axis=1)
    return mask


def _replace_subtractions(expression, modulus):
    assert expression.dim() == 1, expression.shape
    if expression.shape[0] < 2:
        return expression
    mask = (expression == modulus + OP_BY_CHARACTER['-'])
    subtract_replaced = torch.where(mask, modulus + OP_BY_CHARACTER['+'], expression)
    return subtract_replaced[2:] * (1 - 2 * mask[1:-1])


def _perform_multiplications(expression, modulus):
    term_ids = torch.cumsum(expression == modulus + OP_BY_CHARACTER['+'], -1)[::2].long()
    maximum_term_number = expression.shape[0] // 2 + 1
    #print(expression.shape, expression[::2].shape, term_ids.shape)
    products = segment_prod(expression[::2], term_ids, maximum_term_number)
    valid_segment_mask = torch.arange(maximum_term_number) <= term_ids[-1]
    return products * valid_segment_mask


def _replace_blanks(expression, modulus):
    mask = (expression == OP_BY_CHARACTER['_'] + modulus)
    operator_mask = mask
    operator_mask[::2] = False
    residual_mask = mask
    residual_mask[1::2] = False

    blanks_replaced = torch.where(operator_mask, OP_BY_CHARACTER['+'] + modulus, expression)
    blanks_replaced = torch.where(residual_mask, 0, blanks_replaced)
    return blanks_replaced


def _evaluate_expression(expression, modulus):
    expression = _replace_blanks(expression, modulus)
    expression = _replace_subtractions(expression, modulus)
    additive_term = _perform_multiplications(expression, modulus)
    return torch.sum(additive_term, 0) % modulus


OP_BY_CHARACTER = {
    '+': 0,
    '-': 1,
    '*': 2,
    '_': 3
}


def generate_modular_arithmetic_set(B: int, T: int, vocab_size: int):
    modulus = 5
    operators_chars = ('+', '*', '-')
    operators = [OP_BY_CHARACTER[op] for op in operators_chars]

    assert vocab_size == (modulus + len(OP_BY_CHARACTER)), f"vocab size {vocab_size} has to be {modulus + len(OP_BY_CHARACTER)}"

    if T % 2 != 1:
        T -= 1
    batch = torch.empty((B, T), dtype=torch.int)
    remainders = torch.randint(0, modulus, (B, T // 2 + 1))
    
    ops = modulus + torch.Tensor(operators)
    operations = ops[torch.randint(len(ops), (B, T // 2))]
    batch[:, ::2] = remainders
    expressions = batch
    expressions[:, 1::2] = operations

    evaluate = functools.partial(_evaluate_expression, modulus=modulus)
    labels = torch.vmap(evaluate)(expressions)
    return {
        "input_ids": expressions,
        "labels": labels
    }


def _generate_one_expression_and_result(modulus, T, mult=False):
    def gen_terminal():
        #terminal = torch.randint(0, modulus)
        terminal = random.randint(0, modulus-1)
        return str(terminal), terminal
    
    if T < 1:
        raise ValueError(f"Can't generate expressions of length < 1, got {T}")
    if T == 1:
        return gen_terminal()
    elif T == 2:
        term_str, term_val = gen_terminal()
        return f"-{term_str}", -term_val % modulus
    elif T == 3:
        term_str, term_val = gen_terminal()
        return f"({term_str})", term_val % modulus
    elif T == 4:
        term_str, term_val = gen_terminal()
        return f"(-{term_str})", -term_val % modulus
    
    #left_len = torch.randint(1, T-3)
    left_len = random.randint(1, T-4)
    right_len = T - (left_len + 3)
    left_str, left_val = _generate_one_expression_and_result(modulus, left_len, mult)
    right_str, right_val = _generate_one_expression_and_result(modulus, right_len, mult)

    maxop = 3 if mult else 2
    #op = torch.randint(0, maxop)
    op = random.randint(0, maxop)
    if op == 0:
        return "(" + left_str + "+" + right_str + ")", (left_val + right_val) % modulus
    elif op == 1:
        return "(" + left_str + "-" + right_str + ")", (left_val - right_val) % modulus
    else:
        return "(" + left_str + "*" + right_str + ")", (left_val - right_val) % modulus


def _generate_raw_modari_dataset(n, lengths, modulus, mutl=False, with_tqdm=False):
    alphabet_to_int = {
        '+': modulus,
        '-': modulus + 1,
        '*': modulus + 2,
        '(': modulus + 3,
        ')': modulus + 4,
        'x': modulus + 5,
        '=': modulus + 6,
    }
    for x in range(modulus):
        alphabet_to_int[str(x)] = x
    
    def make_default_dict():
        return {
            "expressions": [],
            "results": []
        }
        
    sequences = collections.defaultdict(make_default_dict)
    range_lengths = tqdm.tqdm(lengths) if with_tqdm else lengths
    for length in range_lengths:
        for _ in range(n // len(lengths)):
            seq, labels = _generate_one_expression_and_result(modulus, length, mutl)
            seq = [alphabet_to_int[x] for x in seq]
            sequences[length]["expressions"].append(seq)
            sequences[length]["results"].append(labels)
    #sequences = tree.traverse(
    #    lambda l: torch.Tensor(l),
    #    sequences,
    #    top_down=False
    #)
    return dict(sequences)


def generate_modular_arithmetic_brackets_set(B: int, T: int, vocab_size: int):
    modulus = 5
    mult = True
    with_tqdm=False

    assert vocab_size == modulus + 6, f"vocab size {vocab_size} has to be {modulus + 6}"

    batch = _generate_raw_modari_dataset(B, [T], modulus, mult, with_tqdm)[T]
    return {
        "input_ids": torch.Tensor(batch["expressions"]),
        "labels": torch.Tensor(batch["results"])
    }


def generate_str_modular_arithmetic_set(B: int, T: int, vocab_size: int):
    modulus = 5
    operator_chars = ('+', '-', '*')
    sign_encoding = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '+': 5,
        '-': 6,
        '*': 7,
    }
    
    if T % 2 == 1:
        T -= 1

    def rand_num():
        return str(random.randint(0, 4))
    
    def rand_sign():
        return random.choice(operator_chars)

    equations = []
    labels = []
    for _ in range(B):
        eq = []
        for i in range(0, T-1, 2):
            eq.append(rand_num())
            eq.append(rand_sign())
        eq.append(rand_num())
        l = int(eval("".join(eq))) % modulus
        encode = [sign_encoding[k] for k in eq]
        equations.append(encode)
        labels.append(l)
    equations = torch.Tensor(equations)
    labels = torch.Tensor(labels)
    return {
        "input_ids": equations,
        "labels": labels
    }    