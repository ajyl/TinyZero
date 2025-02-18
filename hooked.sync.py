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

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformer_lens import HookedTransformer
from record_utils import record_activations
from explore_utils import *


# %%


model_path = "checkpoints/TinyZero/v4/actor/global_step_300"
hooked_model_path = "checkpoints/TinyZero/v4/actor/hooked_global_step_300.pt"
data_path = "data/train.parquet"
batch_size = 4
valid_size = 100
max_prompt_length = 256
max_response_length = 150
max_new_tokens = max_response_length
probe_path = "probe_checkpoints/v2/probe.pt"
n_layers = 36
record_module_names = [f"model.layers.{idx}" for idx in range(n_layers)]

# %%

with torch.device("cuda:0"):
    hooked_actor = load_hooked_model(hooked_model_path)

# %%

with torch.device("cuda:1"):
    actor = load_model(model_path)

# %%

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# %%

_, valid_dataloader = get_dataloader(
    data_path, batch_size, max_prompt_length, valid_size, tokenizer
)

# %%

probe = torch.load(probe_path).detach().cuda()

# %%

batch = valid_dataloader.dataset[0]

input_ids = batch["input_ids"].unsqueeze(0).cuda()
attention_mask = batch["attention_mask"].unsqueeze(0).cuda()

generation_config = GenerationConfig(do_sample=False)

# %%

block_size = actor.config.max_position_embeddings
max_new_tokens = 30
orig_output = generate(
    actor,
    input_ids,
    attention_mask,
    max_new_tokens,
    block_size,
    tokenizer.eos_token_id,
)

# %%

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
        logits = tl_model(
            idx_cond,
            return_type="logits",
            attention_mask=attn_mask_cond,
            prepend_bos=True,
            padding_side="left",
            # position_ids=position_ids,
        )
        #logits = output["logits"]
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


max_new_tokens = 200

hooked_output = tl_generate(
    hooked_actor,
    input_ids,
    attention_mask,
    max_new_tokens,
    block_size,
    tokenizer.eos_token_id,
)

# %%

print(tokenizer.batch_decode(orig_output, skip_special_tokens=True))
print(tokenizer.batch_decode(hooked_output, skip_special_tokens=True))
