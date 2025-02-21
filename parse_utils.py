import torch

def get_target_indices(token_ids, target_token, target_next_token, tokenizer):
    """
    Get index of target_token, in which target_next_token shows up after target_token.
    Parameters:

    ----------
    batch : torch.Tensor[batch, seq_len]
    target_token: str
    target_next_token: str
    tokenizer: Tokenizer

    Returns:
    -------
    batch_idxs: torch.Tensor[# matches]
    timestpes: torch.Tensor[# matches]
    """
    target_token_id = tokenizer.encode(target_token)[0]  # 320
    next_token_id = tokenizer.encode(target_next_token)[0]  # 1921

    mask = (token_ids[:, :-1] == target_token_id) & (token_ids[:, 1:] == next_token_id)
    batch_idxs, timesteps = torch.where(mask)
    return batch_idxs, timesteps

