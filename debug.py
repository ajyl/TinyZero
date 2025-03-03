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
import einops
from transformers import (
    GenerationConfig,
)
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import einops
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from record_utils import record_activations, get_module
from HookedQwen import convert_to_hooked_model
from explore_utils import *

# %%

cos = F.cosine_similarity
MASK_TOKEN_ID = 151643


# %%


def unembed_resid_streams(vector, model, k=10):
    """
    vector: [batch, layers, vocab]
    """
    norm = model.model.norm
    lm_head = model.lm_head.weight

    vector = norm(vector)
    dots = einsum(
        "vocab d_model, batch layers d_model -> layers batch vocab", lm_head, vector
    )
    return dots


# %%


def unembed(vector, lm_head, k=10):
    dots = einsum("vocab d_model, d_model -> vocab", lm_head, vector)
    top_k = dots.topk(k).indices
    return top_k


def unembed_text(vector, model, tokenizer, k=10):
    norm = model.model.norm
    lm_head = model.lm_head.weight
    top_k = unembed(vector, lm_head, k=k)
    return tokenizer.batch_decode(top_k, skip_special_tokens=True)


# %%


def remove_all_hooks(model):
    for (
        name,
        module,
    ) in model.named_modules():  # Recursively iterates through submodules
        if hasattr(module, "_forward_hooks"):
            for handle_id in list(module._forward_hooks.keys()):
                module._forward_hooks.pop(handle_id)


def get_all_hooks(model):
    all_hooks = []
    for (
        name,
        module,
    ) in model.named_modules():  # Recursively iterates through submodules
        if hasattr(module, "_forward_hooks"):
            all_hooks.extend(module._forward_hooks.values())

    return all_hooks


# %%

config = {
    "data_path": "data/train.parquet",
    "model_path": "checkpoints/TinyZero/v4/actor/global_step_300",
    "probe_path": "probe_checkpoints/v2/probe.pt",
    "batch_size": 4,
    "valid_size": 256,
    "max_prompt_length": 256,
    "max_response_length": 300,
    "n_layers": 36,
    "d_model": 2048,
    "seed": 42,
}

seed_all(config["seed"], deterministic_algos=True)


# %%

actor = load_model(config["model_path"])
generation_config = GenerationConfig(do_sample=False)
tokenizer = actor.tokenizer

# %%

convert_to_hooked_model(actor)

# %%

_, valid_dataloader = get_dataloader(
    config["data_path"],
    config["batch_size"],
    config["max_prompt_length"],
    config["valid_size"],
    actor.tokenizer,
)

# %%

n_layers = config["n_layers"]
record_module_names = [
    (f"model.layers.{idx}.hook_resid_mid", f"model.layers.{idx}")
    for idx in range(n_layers)
]
record_module_names = [x for sublist in record_module_names for x in sublist]

# %%

token_this = tokenizer.encode("this")[0]  # 574
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921

# %%

#max_new_tokens = 300
#sample_size = 1
#timestep_offset = 1
#samples = []
#for batch_idx, batch in enumerate(valid_dataloader):
#
#    input_ids = batch["input_ids"].cuda()
#    attention_mask = batch["attention_mask"].cuda()
#    with record_activations(actor, record_module_names) as recording:
#        output = actor.generate(
#            input_ids=input_ids,
#            attention_mask=attention_mask,
#            max_new_tokens=max_new_tokens,
#            eos_token_id=tokenizer.eos_token_id,
#            pad_token_id=tokenizer.pad_token_id,
#            generation_config=generation_config,
#            output_scores=False,  # this is potentially very large
#            return_dict_in_generate=True,
#            use_cache=True,
#        )  # may OOM when use_cache = True
#
#    recording = {
#        layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
#    }
#
#    # recording["model.layers.0"].shape:
#    # [batch, prompt_length + max_new_tokens, d_model]
#    seq = output.sequences
#    #response = seq[:, -max_new_tokens:]
#    response = seq
#    response_text = tokenizer.batch_decode(response, skip_special_tokens=True)
#
#    # [batch, n_layers, response_length, d_model]
#    resid_stream = torch.stack(
#        [acts[:, -max_new_tokens:] for acts in recording.values()], dim=1
#    )
#
#    mask_not = (response[:, :-1] == token_open) & (response[:, 1:] == token_not)
#    mask_this = (response[:, :-1] == token_open) & (response[:, 1:] == token_this)
#    batch_idx_not, timesteps_not = torch.where(mask_not)
#    batch_idx_this, timesteps_this = torch.where(mask_this)
#
#    batch_idx_not = batch_idx_not
#    batch_idx_this = batch_idx_this
#
#    overlap_batches = torch.tensor(
#        sorted(
#            list(set(batch_idx_not.tolist()).intersection(set(batch_idx_this.tolist())))
#        )
#    ).cuda()
#    batch_mask_not = torch.isin(batch_idx_not, overlap_batches)
#    batch_mask_this = torch.isin(batch_idx_this, overlap_batches)
#
#    filtered_timesteps_not = {
#        b_idx: timesteps_not[(batch_idx_not == b_idx)]
#        for b_idx in overlap_batches.tolist()
#    }
#    filtered_timesteps_this = {
#        b_idx: timesteps_this[(batch_idx_this == b_idx)]
#        for b_idx in overlap_batches.tolist()
#    }
#
#    for b_idx in filtered_timesteps_this.keys():
#        print("Found a match...")
#        _this_timesteps = filtered_timesteps_this[b_idx]
#        _resid_stream = resid_stream[
#            b_idx,
#            :,
#            _this_timesteps : _this_timesteps + timestep_offset + 1,
#        ]
#        samples.append(
#            {
#                "input_ids": input_ids[b_idx],
#                "attention_mask": attention_mask[b_idx],
#                "resid_stream": _resid_stream,  # [n_layers, timesteps, d_model]
#                "not_timesteps": filtered_timesteps_not[b_idx],
#                "this_timesteps": _this_timesteps,
#                "prompt": tokenizer.batch_decode(
#                    input_ids[b_idx], skip_special_tokens=True
#                ),
#                "output": output.sequences[b_idx],
#            }
#        )
#
#    if len(samples) >= sample_size:
#        break


# %%


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    block_size,
    eos_token_id,
):
    """
    Generate text using a transformer language model with greedy sampling.

    Args:
        model: The auto-regressive transformer model that outputs logits.
        input_ids: A tensor of shape (batch_size, sequence_length) representing the initial token indices.
        max_new_tokens: The number of new tokens to generate.
        block_size: The maximum sequence length (context window) the model can handle.
        device: The device on which computations are performed.

    Returns:
        A tensor containing the original context concatenated with the generated tokens.
    """
    device = "cuda"
    model.eval()  # Set the model to evaluation mode
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    batch_size = input_ids.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in tqdm(range(max_new_tokens)):
        if finished.all():
            break

        if input_ids.shape[1] > block_size:
            idx_cond = input_ids[:, -block_size:]
            attn_mask_cond = attention_mask[:, -block_size:]
        else:
            idx_cond = input_ids
            attn_mask_cond = attention_mask

        position_ids = attn_mask_cond.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask_cond == 0, 1)

        # Get logits from the model. Ensure your model's forward function accepts an attention mask.
        output = model(
            idx_cond,
            attn_mask=attn_mask_cond,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = output["logits"]
        # Focus only on the last time step's logits
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)

        # Greedy sampling: select the token with the highest logit
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)

        new_finished = (~finished) & (next_token.squeeze(1) == eos_token_id)
        finished |= new_finished
        next_token[finished] = eos_token_id

        # Append the predicted token to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        new_mask = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    return input_ids


# %%


@torch.no_grad()
def generate_hooked(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    block_size,
    tokenizer,
    hook_config,
):
    """
    Generate text using a transformer language model with greedy sampling.

    Args:
        model: The auto-regressive transformer model that outputs logits.
        input_ids: A tensor of shape (batch_size, sequence_length) representing the initial token indices.
        max_new_tokens: The number of new tokens to generate.
        block_size: The maximum sequence length (context window) the model can handle.
        device: The device on which computations are performed.

    Returns:
        A tensor containing the original context concatenated with the generated tokens.
    """
    device = "cuda"
    model.eval()  # Set the model to evaluation mode
    remove_all_hooks(model)

    eos_token_id = tokenizer.eos_token_id

    input_ids = input_ids.clone().to(device)
    attention_mask = attention_mask.clone().to(device)
    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Currently only supports a single sample at a time."

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    hook_mlps = hook_config["hook_mlp"]
    hook_target_chars = hook_config["hook_target_chars"]
    hook_timesteps = hook_config["hook_timesteps"]
    hook_token_ids = tokenizer.encode(hook_target_chars)
    print(hook_token_ids)

    hook_batch_idxs = []
    resid_streams = {}
    just_hooked = False
    for timestep in tqdm(range(max_new_tokens)):
        if finished.all():
            break

        if input_ids.shape[1] > block_size:
            idx_cond = input_ids[:, -block_size:]
            attn_mask_cond = attention_mask[:, -block_size:]
        else:
            idx_cond = input_ids
            attn_mask_cond = attention_mask

        position_ids = attn_mask_cond.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask_cond == 0, 1)

        if just_hooked:
            with record_activations(model, record_module_names) as recording:
                output = model(
                    idx_cond,
                    attn_mask=attn_mask_cond,
                    position_ids=position_ids,
                    return_dict=True,
                )
                resid_stream = torch.stack(
                    [acts[0][:, -1] for acts in recording.values()], dim=1
                )
                resid_streams[timestep] = resid_stream
                just_hooked = False
        else:
            output = model(
                idx_cond,
                attn_mask=attn_mask_cond,
                position_ids=position_ids,
                return_dict=True,
            )

        logits = output["logits"]
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)

        most_recent_token = [x[-1] for x in tokenizer.batch_decode(idx_cond)]

        interv_batch_idx = []
        # for batch_idx in range(batch_size):
        # if (
        #    most_recent_token[batch_idx].isdigit()
        #    and next_token[batch_idx].item() in hook_token_ids
        # ):
        #    interv_batch_idx.append(batch_idx)
        # elif most_recent_token[batch_idx] in hook_target_chars:
        #    print("Hooking ...")
        #    interv_batch_idx.append(batch_idx)
        if timestep in hook_timesteps:
            interv_batch_idx = [0]

        if len(interv_batch_idx) > 0:
            print(f"Hooking timestep {timestep}")
            handles = []
            for layer_idx, mlp_idxs in hook_mlps.items():
                handles.append(_add_mlp_hook(actor, layer_idx, mlp_idxs))

            with record_activations(actor, record_module_names) as recording:
                interv_output = model(
                    idx_cond[interv_batch_idx],
                    attn_mask=attn_mask_cond[interv_batch_idx],
                    position_ids=position_ids[interv_batch_idx],
                    return_dict=True,
                )

            # resid_stream: [batch (1), n_layers, d_model]
            resid_stream = torch.stack(
                [acts[0][:, -1] for acts in recording.values()], dim=1
            )
            resid_streams[timestep] = resid_stream
            just_hooked = True

            logits = interv_output["logits"]
            logits = logits[:, -1, :]  # shape: (batch, vocab_size)
            interv_next_token = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # shape: (batch, 1)

            next_token[interv_batch_idx] = interv_next_token

            for handle in handles:
                handle.remove()

        new_finished = (~finished) & (next_token.squeeze(1) == eos_token_id)
        finished |= new_finished
        next_token[finished] = eos_token_id

        # Append the predicted token to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        new_mask = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    return input_ids, resid_streams


# %%

#_input_ids = torch.load("debug.pt")
#_input_ids = _input_ids[:-10]
#_attention_mask = _input_ids != tokenizer.pad_token_id
#_pos_ids = _attention_mask.long().cumsum(-1) - 1
#_pos_ids.masked_fill_(_attention_mask == 0, 1)
#
#orig_output = generate(
#    actor,
#    #samples[0]["input_ids"].unsqueeze(0),
#    _input_ids.unsqueeze(0),
#    #samples[0]["attention_mask"].unsqueeze(0),
#    _attention_mask.unsqueeze(0),
#    300,
#    300,
#    tokenizer.eos_token_id,
#)
#print(tokenizer.batch_decode(orig_output, skip_special_tokens=True))
#breakpoint()

# %%

#_this_t = samples[0]["this_timesteps"]
_this_t = 268
#_input_ids = samples[0]["output"][:_this_t + 1]

_input_ids = torch.load("debug.pt")
_attention_mask = _input_ids != tokenizer.pad_token_id
_pos_ids = _attention_mask.long().cumsum(-1) - 1
_pos_ids.masked_fill_(_attention_mask == 0, 1)


actor.eval()
#with record_activations(actor, record_module_names) as recording:
_out = actor(
    _input_ids.unsqueeze(0),
    attention_mask=_attention_mask.unsqueeze(0),
    position_ids=_pos_ids.unsqueeze(0),
)
next_tok = _out.logits[:, -1].argmax(-1)
print(_out.logits[0, -1][tokenizer.encode("8")])
print(_out.logits[0, -1][tokenizer.encode("6")])
print(tokenizer.decode(next_tok))

#with record_activations(actor, record_module_names) as recording2:
_out2 = actor(
    _input_ids.unsqueeze(0),
    attention_mask=_attention_mask.unsqueeze(0),
    position_ids=_pos_ids.unsqueeze(0),
)
next_tok2 = _out2.logits[:, -1].argmax(-1)
print(_out2.logits[0, -1][tokenizer.encode("8")])
print(_out2.logits[0, -1][tokenizer.encode("6")])
print(tokenizer.decode(next_tok2))
breakpoint()


# %%



# %%
