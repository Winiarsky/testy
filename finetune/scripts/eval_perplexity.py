import argparse
import math
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def compute_perplexity(model, tokenizer, texts, max_length=1024):
    nlls = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids[:, :max_length].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--text_column", default="text")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.adapter_path)

    ds = load_dataset("json", data_files={"validation": args.eval_file})["validation"]
    texts = ds[args.text_column]

    ppl = compute_perplexity(model, tokenizer, texts)
    print({"perplexity": ppl})


if __name__ == "__main__":
    main()
