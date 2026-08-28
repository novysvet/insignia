#!/usr/bin/env python3
"""Dump an official Transformers GLM-5.3 forward pass as an Insignia oracle.

Use the deliberately tiny community checkpoint for fast seam-by-seam parity;
the production checkpoint has the same operators and tensor naming scheme.
"""

import argparse
import pathlib

import numpy as np
import torch
from transformers import Glm5NextForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--tokens", default="1,2,3,4")
    args = parser.parse_args()
    tokens = torch.tensor([[int(token) for token in args.tokens.split(",")]], dtype=torch.long)
    model = Glm5NextForConditionalGeneration.from_pretrained(args.model, dtype="auto").eval()
    captured = {"input_ids": tokens.numpy()}
    hooks = []

    def capture(name):
        def hook(_module, _inputs, output):
            values = output if isinstance(output, tuple) else (output,)
            for index, value in enumerate(values):
                if torch.is_tensor(value):
                    key = name if len(values) == 1 else f"{name}.{index}"
                    captured[key] = value.detach().float().cpu().numpy()
        return hook

    for name, module in model.named_modules():
        wanted = (
            name in {"model.language_model.embed_tokens", "model.language_model.norm", "lm_head"}
            or name.startswith("model.language_model.layers.")
            and name.rsplit(".", 1)[-1] in {
                "attn_hc", "input_layernorm", "self_attn", "ffn_hc",
                "post_attention_layernorm", "mlp",
            }
            or name.count(".") == 3 and name.startswith("model.language_model.layers.")
        )
        if wanted:
            hooks.append(module.register_forward_hook(capture(name)))

    with torch.inference_mode():
        output = model(input_ids=tokens, use_cache=False, output_hidden_states=True)
    for hook in hooks:
        hook.remove()
    captured["logits"] = output.logits.float().cpu().numpy()
    for index, hidden in enumerate(output.hidden_states):
        captured[f"hidden_states.{index}"] = hidden.float().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **captured)
    logits = captured["logits"][0, -1]
    top = np.argpartition(logits, -8)[-8:]
    top = top[np.argsort(logits[top])[::-1]]
    print(f"wrote {args.output} with {len(captured)} arrays")
    print("top8", " ".join(f"{int(index)}:{float(logits[index]):.6f}" for index in top))


if __name__ == "__main__":
    main()
