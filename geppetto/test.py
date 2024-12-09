# coding: utf-8

from __future__ import annotations

from argparse import Namespace

import torch

from geppetto.dataset import load_wiki_data, tokenizer
from geppetto.model import Geppetto


# gpt-3S parameters
params = Namespace(
    # n_context=2048,
    n_context=1024,
    # n_embed=768,
    n_embed=192,
    batch_size=1024,  # 2**19
    n_layers=12,
    n_heads=12,
    attention_bias=False,
    attention_dropout_prob=0.0,
    mlp_factor=4,
    mlp_activation="GELU",
    mlp_dropout_prob=0.0,
    n_vocab=tokenizer.n_vocab,
)

# gpt-3XL parameters
# params = Namespace(
#     n_context=2048,
#     n_embed=2048,
#     batch_size=1024,  # 2**20
#     n_layers=24,
#     n_heads=24,
#     mlp_factor=4,
#     mlp_activation="GELU",
#     dropout_prob=0.0,
#     n_vocab=tokenizer.n_vocab,
# )

# load data
sizes, tensors = load_wiki_data("train", entries=-1)

# dummy batch
x = torch.nn.utils.rnn.pad_sequence(
    [t[:params.n_context] for t in tensors[:32]],
    padding_value=-1,
    batch_first=True,
)

gpt = Geppetto(params).to("cuda:0")
y = gpt(x.to("cuda:0"))
