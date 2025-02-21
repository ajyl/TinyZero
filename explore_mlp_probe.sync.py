# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

import os
import json
import random
from tqdm import tqdm
import numpy as np
from transformers import (
    GenerationConfig,
)
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import einops

from record_utils import record_activations, get_module
from hook_utils import HookWithCountThreshold
from explore_utils import *

# %%

cos = F.cosine_similarity

# %%

config = {
    "data_path": "data/train.parquet",
    "model_path": "checkpoints/TinyZero/v4/actor/global_step_300",
    "probe_path": "probe_checkpoints/probe_from_mlp/probe.pt",
    "batch_size": 64,
    "valid_size": 256,
    "max_prompt_length": 256,
    "max_response_length": 300,
    "n_layers": 36,
    "d_model": 2048,
    "seed": 42,
    "hook_config": {
        "hook_layers": list(range(24, 33)),
        "hook_target_char": " (",
        "hook_target_threshold": 0,
        "hook_scale": 20,
    },
}

seed_all(config["seed"])


# %%

actor = load_model(config["model_path"])
generation_config = GenerationConfig(do_sample=False)

# %%

_, valid_dataloader = get_dataloader(
    config["data_path"],
    config["batch_size"],
    config["max_prompt_length"],
    config["valid_size"],
    actor.tokenizer,
)

# %%

probe_model = torch.load(config["probe_path"]).detach().cuda()
probe_model.shape

# %%


for layer_idx in range(probe_model.shape[0]):
    print(f"Layer {layer_idx}")
    print(cos(probe_model[layer_idx, :, 0], probe_model[layer_idx, :, 1], dim=0))

for layer_idx in range(probe_model.shape[0] - 1):
    print(f"Layer {layer_idx} vs. Layer {layer_idx + 1}")
    print(cos(probe_model[layer_idx, :, 1], probe_model[layer_idx + 1, :, 1], dim=0))

# %%


def get_mlp_value_vecs(model):
    mlp_value_vecs = [layer.mlp.down_proj.weight for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_value_vecs, dim=0)


# %%

value_vecs = get_mlp_value_vecs(actor)
print(value_vecs.shape)

top_cos_scores = {0: [], 1: []}
for target_label in [0, 1]:
    for layer_idx, target_probe_layer in enumerate(range(36)):
        target_probe = probe_model[target_probe_layer, :, target_label]

        print(f"Layer {target_probe_layer}")
        cos_scores = cos(value_vecs[target_probe_layer], target_probe.unsqueeze(-1), dim=0)
        _topk = cos_scores.topk(k=20)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(
            zip(
                _values,
                _idxs,
                [target_probe_layer] * _topk.indices.shape[0],
            )
        )
        top_cos_scores[target_label].extend(topk)

sorted_scores_0 = sorted(top_cos_scores[0], key=lambda x: x[0], reverse=True)
sorted_scores_1 = sorted(top_cos_scores[1], key=lambda x: x[0], reverse=True)


# %%

seen = []
for elem in sorted_scores_0[:100]:
    cos_score, mlp_idx, probe_layer_idx = elem
    curr = (probe_layer_idx, mlp_idx)
    if curr in seen:
        continue
    seen.append(curr)
    print(curr)
    print(cos_score)
    print(
        unembed_text(
            actor.model.layers[probe_layer_idx].mlp.down_proj.weight[:, mlp_idx],
            actor.lm_head.weight,
            actor.tokenizer,
            k=10,
        )
    )

# %%

seen = []
for elem in sorted_scores_1[:100]:
    cos_score, mlp_idx, probe_layer_idx = elem
    curr = (probe_layer_idx, mlp_idx)
    if curr in seen:
        continue
    seen.append(curr)
    print(curr)
    print(probe_layer_idx)
    print(cos_score)
    print(
        unembed_text(
            actor.model.layers[probe_layer_idx].mlp.down_proj.weight[:, mlp_idx],
            actor.lm_head.weight,
            actor.tokenizer,
            k=10,
        )
    )

