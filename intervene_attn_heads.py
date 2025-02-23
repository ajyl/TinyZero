import os
import json
import random
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import einops

from verl.utils.dataset.rl_dataset import RLHFDataset
from record_utils import record_activations, get_module
from hook_utils import HookWithCountThreshold


def seed_all(seed, deterministic_algos=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if deterministic_algos:
        torch.use_deterministic_algorithms()


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


def get_dataloader(config, tokenizer):
    import pandas as pd
    from datasets import Dataset
    from torch.utils.data import DataLoader

    data_path = config["data_path"]
    batch_size = config["batch_size"]
    max_prompt_length = config["max_prompt_length"]
    valid_size = config["valid_size"]
    data = pd.read_parquet(data_path)
    # dataset = Dataset.from_pandas(data)
    dataset = RLHFDataset(
        data_path,
        tokenizer,
        prompt_key="prompt",
        max_prompt_length=max_prompt_length,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
    )

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - valid_size, valid_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def unembed(vector, lm_head, k=10):
    dots = einsum("vocab d_model, d_model -> vocab", lm_head, vector)
    top_k = dots.topk(k).indices
    return top_k


def unembed_text(vector, lm_head, tokenizer, k=10):
    top_k = unembed(vector, lm_head, k=k)
    return tokenizer.batch_decode(top_k, skip_special_tokens=True)


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


@torch.no_grad()
def generate_hooked(
    model,
    input_ids,
    attention_mask,
    steering_vectors,
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
    eos_token_id = tokenizer.eos_token_id

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    batch_size = input_ids.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    hook_target_char = hook_config["hook_target_char"]
    hook_target_threshold = hook_config["hook_target_threshold"]
    hook_scale = hook_config["hook_scale"]
    hook_token_id = tokenizer.encode(hook_target_char)[0]  # 320

    hook_fns = {
        batch_idx: {
            module_name: HookWithCountThreshold(
                steer_vec,
                scale=hook_scale,
                threshold=hook_target_threshold,
                normalize=True,
            )
            for module_name, steer_vec in steering_vectors.items()
        }
        for batch_idx in range(batch_size)
    }

    hook_batch_idxs = []
    open_paran_counter = {batch_idx: 0 for batch_idx in range(batch_size)}
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

        output = model(
            idx_cond,
            attn_mask=attn_mask_cond,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = output["logits"]
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)

        if (next_token == hook_token_id).any():
            hook_idxs = torch.where(next_token == hook_token_id)[0]

            for batch_idx in hook_idxs.tolist():
                open_paran_counter[batch_idx] += 1

        for batch_idx, count in open_paran_counter.items():
            if count < 3:
                continue

            handles = []
            for _module_name, steer_vec in steering_vectors.items():
                hook_fn = hook_fns[batch_idx][_module_name]
                module = get_module(model, _module_name)
                handles.append(module.register_forward_hook(hook_fn))

            interv_output = model(
                idx_cond[hook_idxs],
                attn_mask=attn_mask_cond[hook_idxs],
                position_ids=position_ids[hook_idxs],
                return_dict=True,
            )
            logits = interv_output["logits"]
            logits = logits[:, -1, :]  # shape: (batch, vocab_size)
            interv_next_token = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # shape: (batch, 1)

            next_token[batch_idx] = interv_next_token

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

    return input_ids


def main(config):
    assert torch.cuda.is_available()

    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        # actor_model.to(torch.bfloat16)

    train_dataloader, valid_dataloader = get_dataloader(config, tokenizer)

    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()

    # [vocab, d_model]
    lm_head = actor_model.lm_head.weight

    max_new_tokens = config["max_response_length"]
    max_prompt_length = config["max_prompt_length"]
    n_layers = config["n_layers"]
    d_model = config["d_model"]
    block_size = actor_model.config.max_position_embeddings
    hook_config = config["hook_config"]

    # [n_layers, d_model, 2]
    probe_model = torch.load(config["probe_path"]).cuda()
    steering_vectors = {
        f"model.layers.{idx}": probe_model[idx, :, 1]
        for idx in hook_config["hook_layers"]
    }

    token_open = tokenizer.encode(" (")[0]  # 320
    token_not = tokenizer.encode("not")[0]  # 1921
    token_this = tokenizer.encode("this")[0]  # 574

    for layer_idx in range(n_layers):
        print(f"Layer {layer_idx} ---------------------------------------")
        print(unembed_text(probe_model[layer_idx, :, 0], lm_head, tokenizer, k=10))
        print(unembed_text(probe_model[layer_idx, :, 1], lm_head, tokenizer, k=10))

    for batch_idx, batch in enumerate(valid_dataloader):

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        prompt_len = input_ids.shape[1]
        orig_output = generate(
            actor_model,
            input_ids,
            attention_mask,
            max_new_tokens,
            block_size,
            tokenizer.eos_token_id,
        )
        orig_output_text = tokenizer.batch_decode(
            orig_output[:, prompt_len:], skip_special_tokens=True
        )
        hooked_output = generate_hooked(
            actor_model,
            input_ids,
            attention_mask,
            steering_vectors,
            max_new_tokens,
            block_size,
            tokenizer,
            config["hook_config"],
        )
        interv_output_text = tokenizer.batch_decode(
            hooked_output[:, prompt_len:], skip_special_tokens=True
        )

        print(f"Operands: {batch['nums']}")
        print(f"Targets: {batch['target']}")
        print("Original Output:")
        print(orig_output_text)
        print("Intervened Output:")
        print(interv_output_text)
        print("---------------------------------------")
        print("---------------------------------------")

        breakpoint()

    breakpoint()
    print("z")


if __name__ == "__main__":
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
        "hook_config": {
            "heads": [
                (3, 13),
                (4, 5),
                (4, 0),
                (5, 9),
                (5, 14),
                # (6, 6), (maybe)
                (10, 0),
                (10, 5),
                (11, 8),
                (12, 3),
                (13, 6),
                (13, 3),
                (15, 8),
                (15, 4),
                (17, 14),
                (17, 13),
                (17, 11),
                (17, 10),
                (17, 19),
                (17, 3),
                (17, 1),
                # (18, 7 (maybe)),
                # (18, 3 (maybe)),
                (19, 13),
                (19, 8),
                # (19, 14 (maybe)),
                # (19, 12 (maybe)),
                # (19, 6 (maybe)),
                # (19, 0 (maybe)),
                # (20, 1 (attends to next token after "62")),
                # (20, 3 (attends to next token after "62")),
                # (20, 4 (attends to next token after "62")),
                # (20, 5 (attends to next token after "62")),
                # (20, 6 (attends to next token after "62")),
                (21, 7),
                (21, 14),
                (21, 2),
                (22, 14),
                (22, 12),
                (25, 14),
                (25, 11),
            ],
        },
    }
    seed_all(config["seed"])
    main(config)
