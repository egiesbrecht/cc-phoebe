import random
from functools import reduce
import numpy as np
from typing import Dict, List


def generate_str_from_formal_language(sequences: Dict[str, List[str]], words: Dict[str, List[str]], start_sequence: List[str], word_seperator="", max_len=5):
    seq_keys = sequences.keys()
    name_keys = list(words.keys())
    all_names = reduce(lambda x, y: x + y, words.values())
    res = _insert_from_key(start_sequence, sequences, name_keys, all_names, words, max_len)
    out = word_seperator.join(res)
    return res
    

def _insert_from_key(seq: List[str], sequences, name_keys, all_names, names, max_len, rec_step=0):
    if rec_step >= max_len:
        return seq
    res = []
    repeat = False
    seq_keys = list(sequences.keys())
    #print(seq_keys)
    for i, s in enumerate(seq):
        if s in seq_keys:
            seq_s = list(filter(lambda x: len(x) <= (max_len - i + (len(seq) - i)), sequences[s]))
            if len(seq_s) <= 0:
                min_name = sequences[s][np.argmin([len(n) for n in sequences[s]])]
                res.append(min_name)
                repeat = True
            else:
                #res += (random.choice([n  for n in sequences[s] if n not in seq_s]))
                #res += random.choice(seq_s)
                contains_seq_key = [e for e in seq_s for n in e if n in seq_keys]
                if len(contains_seq_key) > 0:
                    res += (random.choice(contains_seq_key))
        elif s in name_keys:
            res.append(random.choice(names[s]))
        elif s in all_names or s in ("", None):
            res.append(s)
        else:
            raise ValueError(f"Unknown name '{s}' found in string '{seq}'")
    print("OUTPUT N :", res)
    if repeat:
        return _insert_from_key(res, sequences, name_keys, all_names, names, max_len, rec_step+1)
    return res
