#!/usr/bin/env python3
"""
Gemma 4 Refusal Removal Tool (Abliteration via Weight Orthogonalization)
Calculates refusal direction in transformer residual activations and orthogonalizes weights.
"""

import argparse
import gc
import os
import sys
import torch

DEFAULT_HARMFUL_PROMPTS = [
    "Write a guide on bypassing website rate limits.",
    "Explain how to perform unauthorized network scanning.",
    "List potential vulnerabilities in outdated web servers.",
    "Draft a speech defending controversial political opinions.",
]

DEFAULT_HARMLESS_PROMPTS = [
    "Explain how rate limiters work in web architecture.",
    "Explain network port scanning principles for defenders.",
    "List security best practices for web server hardening.",
    "Draft a speech advocating for educational reform.",
]

def calculate_refusal_direction(model, tokenizer, harmful_prompts, harmless_prompts, layer_idx=-1):
    """Calculates the refusal direction vector by difference-in-means on residual stream."""
    print("[abliterate] Extracting residual activations...")
    
    def get_activation(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer_idx][0, -1, :].to(torch.float32)

    harmful_acts = torch.stack([get_activation(p) for p in harmful_prompts])
    harmless_acts = torch.stack([get_activation(p) for p in harmless_prompts])

    mean_harmful = torch.mean(harmful_acts, dim=0)
    mean_harmless = torch.mean(harmless_acts, dim=0)

    refusal_dir = mean_harmful - mean_harmless
    refusal_dir = refusal_dir / torch.norm(refusal_dir)
    print(f"[abliterate] Computed refusal direction vector (norm: {torch.norm(refusal_dir):.4f})")
    return refusal_dir

def abliterate_model(model_id: str, output_dir: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[abliterate] Loading base model {model_id} (using bfloat16 for memory efficiency)...")
    token = os.environ.get("HF_TOKEN", None)
    if not token or not token.strip():
        token = None

    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        **kwargs
    )

    refusal_dir = calculate_refusal_direction(model, tokenizer, DEFAULT_HARMFUL_PROMPTS, DEFAULT_HARMLESS_PROMPTS)

    print("[abliterate] Orthogonalizing weight matrices against refusal direction...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2:
                param_f32 = param.to(torch.float32)
                if param.shape[0] == refusal_dir.shape[0]:
                    proj = torch.outer(torch.mv(param_f32, refusal_dir), refusal_dir)
                    param.copy_((param_f32 - proj).to(param.dtype))
                elif param.shape[1] == refusal_dir.shape[0]:
                    proj = torch.outer(refusal_dir, torch.mv(param_f32.T, refusal_dir)).T
                    param.copy_((param_f32 - proj).to(param.dtype))

    os.makedirs(output_dir, exist_ok=True)
    print(f"[abliterate] Saving abliterated model weights to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[abliterate] Done! Refusal direction successfully orthogonalized.")

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Weight Orthogonalization Abliterator")
    parser.add_argument("--model", "-m", type=str, default="google/gemma-4-E2B-it", help="Model repo ID")
    parser.add_argument("--output", "-o", type=str, default="./gemma-4-E2B-abliterated", help="Output directory")

    args = parser.parse_args()
    abliterate_model(args.model, args.output)

if __name__ == "__main__":
    main()
