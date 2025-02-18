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
    "probe_path": "probe_checkpoints/v2/probe.pt",
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
    for target_probe_layer in range(24, 36):
        target_probe = probe_model[target_probe_layer, :, target_label]

        for layer_idx in range(0, target_probe_layer + 1):
            print(f"Layer {layer_idx}")
            cos_scores = cos(value_vecs[layer_idx], target_probe.unsqueeze(-1), dim=0)
            _topk = cos_scores.topk(k=100)
            _values = [x.item() for x in _topk.values]
            _idxs = [x.item() for x in _topk.indices]
            topk = list(
                zip(
                    _values,
                    _idxs,
                    [target_probe_layer] * _topk.indices.shape[0],
                    [layer_idx] * _topk.indices.shape[0],
                )
            )
            top_cos_scores[target_label].extend(topk)

sorted_scores_0 = sorted(top_cos_scores[0], key=lambda x: x[0], reverse=True)
sorted_scores_1 = sorted(top_cos_scores[1], key=lambda x: x[0], reverse=True)


# %%

seen = []
for elem in sorted_scores_0[:100]:
    cos_score, mlp_idx, probe_layer_idx, layer_idx = elem
    curr = (layer_idx, mlp_idx)
    if curr in seen:
        continue
    seen.append(curr)
    print(curr)
    print(cos_score)
    print(
        unembed_text(
            actor.model.layers[layer_idx].mlp.down_proj.weight[:, mlp_idx],
            actor.lm_head.weight,
            actor.tokenizer,
            k=10,
        )
    )

# %%

seen = []
for elem in sorted_scores_1[:100]:
    cos_score, mlp_idx, probe_layer_idx, layer_idx = elem
    curr = (layer_idx, mlp_idx)
    if curr in seen:
        continue
    seen.append(curr)
    print(curr)
    print(probe_layer_idx)
    print(cos_score)
    print(
        unembed_text(
            actor.model.layers[layer_idx].mlp.down_proj.weight[:, mlp_idx],
            actor.lm_head.weight,
            actor.tokenizer,
            k=10,
        )
    )

# %%

print(
    cos(
        actor.model.layers[25].mlp.down_proj.weight[:, 7613],
        actor.model.layers[25].mlp.down_proj.weight[:, 1688],
        dim=0,
    )
)

print(
    cos(
        actor.model.layers[25].mlp.down_proj.weight[:, 7613],
        probe_model[25, :, 1],
        dim=0,
    )
)

print(
    cos(
        actor.model.layers[25].mlp.down_proj.weight[:, 1688],
        probe_model[25, :, 1],
        dim=0,
    )
)

print(
    cos(
        actor.model.layers[25].mlp.down_proj.weight[:, 1688]
        + actor.model.layers[25].mlp.down_proj.weight[:, 7613],
        probe_model[25, :, 1],
        dim=0,
    )
)

# %%

mlp_idxs = [1688, 7613, 9748, 3521, 2929, 8947]

for offset in range(1, len(mlp_idxs)):
    added = actor.model.layers[25].mlp.down_proj.weight[:, mlp_idxs[:offset]].sum(dim=1)

    print(
        unembed_text(
            added,
            # actor.model.layers[25].mlp.down_proj.weight[:, 1688]
            # + actor.model.layers[25].mlp.down_proj.weight[:, 7613],
            actor.lm_head.weight,
            actor.tokenizer,
            k=10,
        )
    )
    print(cos(added, probe_model[25, :, 1], dim=0))

# [28, 10153]
# [29, 6676]

# %%


# %%

mlp_layers = list(range(24, 36))
record_module_names = [f"model.layers.{i}.mlp.act_fn" for i in mlp_layers]
max_new_tokens = config["max_response_length"]
tokenizer = actor.tokenizer
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921
token_this = tokenizer.encode("this")[0]  # 574


not_acts = []
this_acts = []
for batch_idx, batch in enumerate(valid_dataloader):

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    with record_activations(actor, record_module_names) as recording:
        output = actor.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=generation_config,
            output_scores=False,  # this is potentially very large
            return_dict_in_generate=True,
            use_cache=True,
        )

    # len(recording["model.layers.0"]): max_response_length
    # recording["model.layers.0"][0].shape: [batch, prompt_length, d_model]
    # recording["model.layers.0"][1].shape: [batch, 1, d_model]
    recording = {
        layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
    }

    # recording["model.layers.0"].shape:
    # [batch, prompt_length + max_new_tokens, d_mlp]
    seq = output.sequences
    response = seq[:, -max_new_tokens:]
    response_text = tokenizer.batch_decode(seq, skip_special_tokens=True)

    # [batch, n_layers, response_length, d_mlp]
    activations = torch.stack(
        [acts[:, -max_new_tokens:] for acts in recording.values()], dim=1
    )
    print(resid_stream.shape)

    mask_not = (response[:, :-1] == token_open) & (response[:, 1:] == token_not)
    mask_this = (response[:, :-1] == token_open) & (response[:, 1:] == token_this)
    batch_idx_not, timesteps_not = torch.where(mask_not)
    batch_idx_this, timesteps_this = torch.where(mask_this)

    batch_idx_not = batch_idx_not
    batch_idx_this = batch_idx_this

    overlap_batches = torch.tensor(
        sorted(
            list(set(batch_idx_not.tolist()).intersection(set(batch_idx_this.tolist())))
        )
    ).cuda()
    batch_mask_not = torch.isin(batch_idx_not, overlap_batches)
    batch_mask_this = torch.isin(batch_idx_this, overlap_batches)

    # TODO: probe_timestep_offset.
    filtered_timesteps_not = {
        b_idx: timesteps_not[(batch_idx_not == b_idx)]
        for b_idx in overlap_batches.tolist()
    }
    filtered_timesteps_this = {
        b_idx: timesteps_this[(batch_idx_this == b_idx)]
        for b_idx in overlap_batches.tolist()
    }

    for b_idx in filtered_timesteps_not.keys():
        _not_timesteps = filtered_timesteps_not[b_idx].tolist()
        not_acts.append(
            activations[
                b_idx,
                :,
                _not_timesteps,
            ].cpu()
        )
        _this_timesteps = filtered_timesteps_this[b_idx].tolist()
        this_acts.append(
            activations[
                b_idx,
                :,
                _this_timesteps,
            ].cpu()
        )

not_acts = torch.cat(not_acts, dim=1)
this_acts = torch.cat(this_acts, dim=1)

# %%

_not_acts = not_acts.clone()
_this_acts = this_acts.clone()

# [_layers, d_mlp]
_not_acts = _not_acts.mean(dim=1).cuda()
_this_acts = _this_acts.mean(dim=1).cuda()

_value_vecs = value_vecs[mlp_layers]

print(_not_acts.shape)
print(_value_vecs.shape)

# [layers, d_model, d_mlp]
neg_scaled_value_vecs = _not_acts.unsqueeze(1) * _value_vecs
pos_scaled_value_vecs = _this_acts.unsqueeze(1) * _value_vecs

# [layers, d_mlp]
dot_prods = einsum(
    "layers d_model d_mlp, layers d_model -> layers d_mlp",
    pos_scaled_value_vecs,
    probe_model[mlp_layers, :, 1],
)

for layer_idx in range(dot_prods.shape[0]):
    top_idxs = dot_prods[layer_idx].topk(k=10).indices

    curr_layer = mlp_layers[layer_idx]
    for _idx in top_idxs.tolist():
        print(f"Layer {curr_layer} Index {_idx}")
        curr_value_vecs = actor.model.layers[curr_layer].mlp.down_proj.weight[:, _idx]

        print(unembed_text(curr_value_vecs, actor.lm_head.weight, actor.tokenizer, k=10))
