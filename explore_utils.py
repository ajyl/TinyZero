import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from fancy_einsum import einsum
from verl.utils.dataset.rl_dataset import RLHFDataset


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


def load_model(model_path):
    assert torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
    actor_model.tokenizer = tokenizer
    return actor_model


def _collate_fn(data_list: list[dict]) -> dict:
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




def get_dataloader(data_path, batch_size, max_prompt_length, valid_size, tokenizer):

    data = pd.read_parquet(data_path)
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
        collate_fn=_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    return train_loader, valid_loader


def unembed_text(vector, lm_head, tokenizer, k=10):
    top_k = unembed(vector, lm_head, k=k)
    return tokenizer.batch_decode(top_k, skip_special_tokens=True)


def unembed(vector, lm_head, k=10):
    dots = einsum("vocab d_model, d_model -> vocab", lm_head, vector)
    top_k = dots.topk(k).indices
    return top_k


def load_hooked_model(model_path):
    hf_config = {
        "_attn_implementation_autoset": True,
        "_name_or_path": "Qwen/Qwen2.5-3B",
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 32768,
        "max_window_layers": 36,
        "model_type": "qwen2",
        "num_attention_heads": 16,
        "num_hidden_layers": 36,
        "num_key_value_heads": 2,
        "pad_token_id": 151643,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.47.0",
        "use_cache": True,
        "use_mrope": False,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }
    tf_cfg = get_pretrained_model_config("Qwen/Qwen2.5-3B", hf_cfg=hf_config, fold_ln=False)
    state_dict = torch.load(model_path)
    hooked_model = HookedTransformer(tf_cfg)
    hooked_model.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    #tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    hooked_model.tokenizer = tokenizer
    return hooked_model


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
def tl_generate(
    tl_model,
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
    tl_model.eval()  # Set the model to evaluation mode
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
        output = tl_model(
            idx_cond,
            attention_mask=attn_mask_cond,
            # position_ids=position_ids,
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

