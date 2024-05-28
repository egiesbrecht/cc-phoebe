import glob
import pandas as pd
from datasets import Dataset
import numpy as np


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
