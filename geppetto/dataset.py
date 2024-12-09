# coding: utf-8

from __future__ import annotations

import os
import pickle
import multiprocessing

import numpy as np
import torch
import datasets  # type: ignore[import-untyped]
import tiktoken
import spacy
import tqdm  # type: ignore[import-untyped]


cache_dir = os.path.join(os.environ["GP_DATA_BASE"], "cache")

# wikipedia dump
wiki_dump = "20220301.en"

# gpt-2/3 tokenizer
tokenizer = tiktoken.encoding_for_model("gpt2")
EOT = tokenizer.eot_token

# document parser for sentence splitting
doc_parser = spacy.load(os.environ["GP_SPACY_MODEL_EN"])


def split_sentences(text: str, /) -> list[str]:
    doc = doc_parser(text)
    return list(map(str, doc.sents))


def tokenize_(index: int, text: str, /) -> tuple[int, list[int]]:
    return index, tokenizer.encode(text)


def tokenize_mp_(args, /) -> tuple[int, list[int]]:
    return tokenize_(*args)


def load_wiki_data(
    split: str,
    /,
    *,
    entries: int = -1,
    mp_processes: int = 20,
    mp_chunksize: int = 1000,
    update_cache: bool = False,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    # check if cached
    cache_parts = [wiki_dump.replace(".", ""), split]
    if entries >= 0:
        cache_parts.append(str(entries))
    cache_path = os.path.join(cache_dir, f"wiki_{'_'.join(cache_parts)}.pkl")
    if os.path.exists(cache_path):
        if not update_cache:
            print(f"loading tokens from cached {cache_path}")
            with open(cache_path, "rb") as fr:
                return pickle.load(fr)
        os.remove(cache_path)

    # load wiki dataset and preprocess
    wiki = datasets.load_dataset("wikipedia", wiki_dump, split=split, trust_remote_code=True, num_proc=mp_processes)
    entries = len(wiki) if entries < 0 else min(entries, len(wiki))

    # compute sizes and convert to tensors
    tensors: list[torch.Tensor] = entries * [torch.empty(0)]
    with multiprocessing.Pool(processes=mp_processes) as pool:
        print(f"preparing {entries:_} documents for tokenization with {mp_processes} processes")
        gen = ((i, wiki[i]["text"]) for i in range(entries))
        try:
            res = pool.imap_unordered(tokenize_mp_, gen, chunksize=mp_chunksize)
            for i, tokens in tqdm.tqdm(res, desc="tokenizing inputs", total=entries):
                tensors[i] = torch.tensor(tokens)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
    sizes = np.array([len(t) for t in tensors])

    # save to cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"writing tokens to cache {cache_path}")
    with open(cache_path, "wb") as fw:
        pickle.dump((sizes, tensors), fw)

    return sizes, tensors
