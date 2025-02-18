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
    actor = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# %%

hf_config = {
    "_attn_implementation_autoset": True,
    "_name_or_path": "checkpoints/TinyZero/v4/actor/global_step_300",
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

# %%

from transformer_lens.pretrained.weight_conversions.qwen2 import convert_qwen2_weights
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

tf_cfg = get_pretrained_model_config("Qwen/Qwen2.5-3B", hf_cfg=hf_config, fold_ln=False)

state_dict = convert_qwen2_weights(actor, tf_cfg)

# %%

torch.save(state_dict, "checkpoints/TinyZero/v4/actor/hooked_global_step_300.pt")
