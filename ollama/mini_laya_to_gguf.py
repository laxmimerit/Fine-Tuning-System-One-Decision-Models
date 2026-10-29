"""Write a trained Mini Laya folder as one GGUF file that Ollama runs as a decision model.

Mini Laya is laya's decision head on ModernBERT-base, the same machine as Laya on ModernBERT-large,
so it is written in the layout of Ollama's own laya build: architecture modern-bert, decision
type laya. Everything comes from the Mini Laya folder itself: the weights, the encoder's config,
the tokenizer and the fitted temperatures.

usage: python mini_laya_to_gguf.py <mini-laya folder> <out .gguf>
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize
from safetensors import safe_open

# The prompt layout Laya's decision head reads: the question, one [MASK] per option, then the text.
SYSTEMONE_TEMPLATE = (
    "[CLS]{{ type }} question: {{ instructions if instructions is string else instructions | tojson }}[SEP]"
    "{% for o in options %}[MASK] {% if type == 'choice' %}{{ o.key }}{% if o.description %}: "
    "{{ o.description if o.description is string else o.description | tojson }}{% endif %}"
    "{% elif type == 'score' %}level {{ o.key }}: "
    "{{ o.description if o.description is string else o.description | tojson }}"
    "{% else %}{{ o.key }}: {% if o.description %}"
    "{{ o.description if o.description is string else o.description | tojson }}"
    "{% elif o.key == 'true' %}yes, the statement holds{% else %}no, the statement does not hold{% endif %}"
    "{% endif %}{% endfor %}[SEP]{{ state if state is string else state | tojson }}[SEP]"
)

src, out = Path(sys.argv[1]), sys.argv[2]
cfg = json.loads((src / "rl_agent_config.json").read_text(encoding="utf-8"))  # head settings, temperatures
enc = json.loads((src / "encoder" / "config.json").read_text(encoding="utf-8"))  # ModernBERT's settings
tok = json.loads((src / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))

st = safe_open(src / "model.safetensors", "np")
w = {k: st.get_tensor(k).astype(np.float32) for k in st.keys()}
d, n_enc, n_head = enc["hidden_size"], enc["num_hidden_layers"], cfg["head_layers"]

g = GGUFWriter(out, "modern-bert")
g.add_type("model")
g.add_name("Mini Laya")
g.add_license("apache-2.0")
g.add_block_count(n_enc + n_head)  # the encoder's layers, then the head's
g.add_context_length(enc["max_position_embeddings"])
g.add_embedding_length(d)
g.add_feed_forward_length([enc["intermediate_size"]] * n_enc + [4 * d] * n_head)
g.add_head_count(enc["num_attention_heads"])
g.add_rope_freq_base(enc["rope_parameters"]["full_attention"]["rope_theta"])
g.add_float32("modern-bert.rope.freq_base_swa", enc["rope_parameters"]["sliding_attention"]["rope_theta"])
g.add_layer_norm_rms_eps(enc["norm_eps"])
g.add_layer_norm_eps(enc["norm_eps"])
g.add_file_type(32)  # BF16
g.add_causal_attention(False)  # an encoder reads the whole text at once
g.add_uint32("modern-bert.classifier.pooling_type", 1)
g.add_string("modern-bert.classifier.activation", enc["classifier_activation"])
g.add_sliding_window(enc["local_attention"])
g.add_uint32("modern-bert.attention.sliding_window_pattern", enc["global_attn_every_n_layers"])
g.add_string("modern-bert.rope.scaling.type", "none")
g.add_vocab_size(enc["vocab_size"])
g.add_string("modern-bert.hidden_activation", enc["hidden_activation"])

# What makes Ollama treat it as a decision model.
g.add_string("modern-bert.decision.type", "laya")
g.add_uint32("modern-bert.decision.block_count", n_head)
g.add_uint32("modern-bert.decision.max_head_tokens", cfg["head_max_len"])
for qtype, t in zip(("choice", "score", "noul"), cfg["temperature"]):
    g.add_float32(f"modern-bert.decision.temperature.{qtype}", float(t))
g.add_quantization_version(2)

# The tokenizer, every token by its id; added tokens are control (special) or user defined.
tokens = [""] * enc["vocab_size"]
types = [1] * enc["vocab_size"]
for t, i in tok["model"]["vocab"].items():
    tokens[i] = t
for a in tok["added_tokens"]:
    tokens[a["id"]], types[a["id"]] = a["content"], 3 if a["special"] else 4
g.add_tokenizer_model("gpt2")
g.add_tokenizer_pre("modern-bert")
g.add_token_list(tokens)
g.add_token_types(types)
g.add_token_merges([" ".join(m) for m in tok["model"]["merges"]])
g.add_bool("tokenizer.ggml.add_bos_token", True)
g.add_bool("tokenizer.ggml.add_eos_token", True)
g.add_bool("tokenizer.ggml.add_sep_token", False)
ids = {a["content"]: a["id"] for a in tok["added_tokens"]}
for key, name in (("bos", "[CLS]"), ("eos", "[SEP]"), ("unknown", "[UNK]"), ("seperator", "[SEP]"),
                  ("padding", "[PAD]"), ("mask", "[MASK]")):
    g.add_uint32(f"tokenizer.ggml.{key}_token_id", ids[name])
g.add_uint32("tokenizer.ggml.token_type_count", 3)
g.add_string("tokenizer.chat_template.systemone", SYSTEMONE_TEMPLATE)
g.add_array("tokenizer.chat_templates", ["systemone"])


def put(name, x):
    if x.ndim == 1 or name == "token_types.weight":  # added to f32 activations, so kept f32 as in laya
        g.add_tensor(name, x.astype(np.float32))
    else:
        g.add_tensor(name, quantize(x, GGMLQuantizationType.BF16), raw_dtype=GGMLQuantizationType.BF16)


# The encoder, layer by layer.
put("token_embd.weight", w["encoder.embeddings.tok_embeddings.weight"])
put("token_embd_norm.weight", w["encoder.embeddings.norm.weight"])
for i in range(n_enc):
    p = f"encoder.layers.{i}."
    if p + "attn_norm.weight" in w:  # layer 0 has none
        put(f"blk.{i}.attn_norm.weight", w[p + "attn_norm.weight"])
    put(f"blk.{i}.attn_qkv.weight", w[p + "attn.Wqkv.weight"])
    put(f"blk.{i}.attn_output.weight", w[p + "attn.Wo.weight"])
    put(f"blk.{i}.ffn_norm.weight", w[p + "mlp_norm.weight"])
    put(f"blk.{i}.ffn_up.weight", w[p + "mlp.Wi.weight"])
    put(f"blk.{i}.ffn_down.weight", w[p + "mlp.Wo.weight"])
put("output_norm.weight", w["encoder.final_norm.weight"])

# The decision head: its layers numbered after the encoder's, then the type embedding and the scorer.
names = {"self_attn.in_proj_": "attn_qkv.", "self_attn.out_proj.": "attn_output.", "norm1.": "attn_norm.",
         "norm2.": "ffn_norm.", "linear1.": "ffn_up.", "linear2.": "ffn_down."}
for j in range(n_head):
    for a, b in names.items():
        for part in ("weight", "bias"):
            put(f"blk.{n_enc + j}.{b}{part}", w[f"head.layers.{j}.{a}{part}"])
put("token_types.weight", w["type_emb.weight"])
for a, b in (("scorer.0.", "cls.norm."), ("scorer.1.", "cls."), ("scorer.3.", "cls.output.")):
    for part in ("weight", "bias"):
        put(f"{b}{part}", w[f"{a}{part}"])

g.write_header_to_file()
g.write_kv_data_to_file()
g.write_tensors_to_file()
g.close()
print("wrote", out, f"{Path(out).stat().st_size / 1024**2:.1f} MB")
