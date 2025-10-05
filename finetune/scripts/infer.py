import os
import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel


def load_model_and_tokenizer(base_model_path: str, adapter_path: Optional[str] = None):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.adapter_path)

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = args.prompt or "Hello"

    text = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
    print("\n---\n", text)


if __name__ == "__main__":
    main()
