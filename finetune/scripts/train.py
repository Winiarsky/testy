import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_bnb_config(cfg: Dict[str, Any]):
    if not cfg.get("load_in_4bit", False):
        return None
    try:
        import bitsandbytes as bnb  # noqa: F401
        from transformers import BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes not available but load_in_4bit=True; install correctly or set to False"
        ) from e

    compute_dtype = getattr(torch, str(cfg.get("bnb_4bit_compute_dtype", "float16")))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
    )
    return bnb_config


def prepare_tokenizer(cfg: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_path"],
        use_fast=True,
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_examples(examples: Dict[str, List[str]], text_column: str) -> Dict[str, List[str]]:
    return {"text": examples[text_column]}


def tokenize_function(examples, tokenizer, max_seq_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_attention_mask=False,
    )


def maybe_pack_dataset(tokenized_ds, chunk_size: int):
    # Group Texts for efficient causal LM training
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_ds.map(group_texts, batched=True)


def build_lora(cfg: Dict[str, Any]):
    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("DISABLE_WANDB", "true")

    bnb_config = get_bnb_config(cfg)
    torch_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_path"],
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.get("trust_remote_code", True),
        quantization_config=bnb_config,
    )

    tokenizer = prepare_tokenizer(cfg)

    # Data
    train_file = cfg.get("train_file")
    eval_file = cfg.get("eval_file")
    text_column = cfg.get("text_column", "text")

    data_files = {}
    if train_file:
        data_files["train"] = train_file
    if eval_file:
        data_files["validation"] = eval_file

    raw_datasets = load_dataset("json", data_files=data_files)
    raw_datasets = raw_datasets.map(
        lambda ex: format_examples(ex, text_column), batched=True, remove_columns=[text_column]
    )

    tokenized = raw_datasets.map(
        lambda ex: tokenize_function(ex, tokenizer, int(cfg.get("max_seq_length", 1024))),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    if cfg.get("packing", True):
        tokenized["train"] = maybe_pack_dataset(tokenized["train"], int(cfg.get("max_seq_length", 1024)))
        if "validation" in tokenized:
            tokenized["validation"] = maybe_pack_dataset(tokenized["validation"], int(cfg.get("max_seq_length", 1024)))
    else:
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

    # LoRA
    if cfg.get("use_lora", True):
        lora_config = build_lora(cfg)
        model = get_peft_model(model, lora_config)

    # Training args
    output_dir = cfg.get("output_dir", "outputs/run-qlora")
    logging_steps = int(cfg.get("logging_steps", 10))
    eval_steps = int(cfg.get("eval_steps", 100))
    save_steps = int(cfg.get("save_steps", 200))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=logging_steps,
        evaluation_strategy=str(cfg.get("evaluation_strategy", "steps")),
        eval_steps=eval_steps,
        save_steps=save_steps,
        bf16=bool(cfg.get("bf16", True)),
        fp16=bool(cfg.get("fp16", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        report_to=cfg.get("report_to", "none"),
        save_total_limit=cfg.get("save_total_limit", 2),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter or full model
    adapter_dir = cfg.get("adapter_dir")
    if cfg.get("use_lora", True):
        save_dir = adapter_dir or os.path.join(output_dir, "adapter")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(json.dumps({"saved_adapter": save_dir}))
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(json.dumps({"saved_model": output_dir}))


if __name__ == "__main__":
    main()
