# coding: utf-8

from __future__ import annotations

from argparse import Namespace

import torch
import torchinfo

from geppetto.dataset import load_wiki_data, tokenizer
from geppetto.model import Geppetto


device = torch.device("cuda:0")

# GPT-3 "mini" parameters
params = Namespace(
    n_context=1024,
    n_embed=192,
    batch_size=512,
    n_layers=6,
    n_heads=12,
    attention_bias=False,
    attention_dropout_prob=0.05,
    mlp_factor=4,
    mlp_activation="GELU",
    mlp_bias=True,
    mlp_dropout_prob=0.05,
    n_vocab=tokenizer.n_vocab,
    embed_dtype=torch.float16,
    qkv_dtype=torch.float16,
    softmax_dtype=torch.float32,
    proj_dtype=torch.float16,
    mlp_dtype=torch.float16,
    norm_dtype=torch.float32,
)

# # GPT-3 S changes
# params.n_context = 2048
# params.n_embed = 768
# params.n_heads = 12
# params.n_layers = 12

# # GPT-3 XL changes
# params.n_context = 2048
# params.n_embed = 2048
# params.n_heads = 16  # the paper quotes 24 which is a typo :)
# params.n_layers = 24

# # GPT-3 175B changes
# params.n_context = 2048
# params.n_embed = 12288
# params.n_heads = 96
# params.n_layers = 96

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
sizes, tensors = load_wiki_data("train", entries=10_000)

# dummy batch
x = torch.nn.utils.rnn.pad_sequence(
    [t[:params.n_context] for t in tensors[:32]],
    padding_value=-1,
    batch_first=True,
)
gpt = Geppetto(params)

x = x.to(device)
gpt = gpt.to(device)
# y = gpt(x.to("cuda:0"))

with torch.autocast(device_type=device.type, enabled=True):
    print(torchinfo.summary(gpt, input_data=(x,)))
    from IPython import embed; embed(header="debugger")
