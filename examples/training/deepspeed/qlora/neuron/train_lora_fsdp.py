# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""bf16 LoRA fine-tuning of Qwen3-8B on AWS Trainium: the Neuron alternative to the QLoRA example.

Same model, dataset, prompt format and LoRA settings as ../src/train.py, but:

- bf16 weights instead of bitsandbytes NF4 (bitsandbytes has no Neuron backend), sharded across
  NeuronCores with PyTorch FSDP2, instead of DeepSpeed ZeRO.
- PyTorch AdamW instead of paged_adamw_8bit.
- Sequences padded to a fixed length so every step reuses the same compiled graphs.

Launch with torchrun (one process per logical NeuronCore):

    torchrun --nproc_per_node=4 neuron/train_lora_fsdp.py --config_path configs/training_config_zero2.yaml
"""

import argparse
import logging
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import FullConfig  # noqa: E402
from data_preparation import load_medical_dataset, preprocess_dataset  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_path", default="configs/training_config_zero2.yaml")
    p.add_argument("--max_steps", type=int, default=-1, help="Override the epoch count for a short run")
    p.add_argument("--max_samples", type=int, default=None, help="Limit the dataset for a quick test")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def pad_collator(tokenizer, max_length):
    """Pad every batch to max_length (static shapes for the Neuron compiler)."""
    def collate(features):
        ids = torch.full((len(features), max_length), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)
        mask = torch.zeros((len(features), max_length), dtype=torch.long)
        for i, f in enumerate(features):
            n = min(len(f["input_ids"]), max_length)
            ids[i, :n] = torch.tensor(f["input_ids"][:n])
            labels[i, :n] = torch.tensor(f["labels"][:n])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return collate


def main():
    args = parse_args()
    config = FullConfig.from_yaml(args.config_path)
    m, q, d, t = config.model, config.qlora, config.data, config.training

    tokenizer = AutoTokenizer.from_pretrained(m.model_name, trust_remote_code=m.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw, eval_raw = load_medical_dataset(
        dataset_name=d.dataset_name, config_name=d.dataset_config,
        max_samples=args.max_samples or d.max_samples, train_split=d.train_split)
    train_ds = preprocess_dataset(train_raw, tokenizer, max_length=m.max_seq_length)
    eval_ds = preprocess_dataset(eval_raw, tokenizer, max_length=m.max_seq_length)

    # bf16 weights; FSDP2 shards them across ranks. attn_implementation can be overridden with ATTN_IMPL.
    model = AutoModelForCausalLM.from_pretrained(
        m.model_name, dtype=torch.bfloat16, trust_remote_code=m.trust_remote_code,
        attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"))
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=q.lora_r, lora_alpha=q.lora_alpha, lora_dropout=q.lora_dropout, bias=q.bias,
        task_type=q.task_type, target_modules=list(q.target_modules)))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir or t.output_dir,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=t.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_steps=t.warmup_ratio,  # a value < 1 is a fraction of the total steps
        lr_scheduler_type=t.lr_scheduler_type,
        optim="adamw_torch",
        max_grad_norm=t.max_grad_norm,
        bf16=True,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        eval_steps=t.eval_steps,
        eval_strategy="steps",
        save_total_limit=t.save_total_limit,
        report_to=t.report_to,
        seed=t.seed,
        gradient_checkpointing=config.parallel.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fsdp="full_shard auto_wrap",
        fsdp_config={"version": 2, "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
                     "reshard_after_forward": True, "cpu_ram_efficient_loading": False},
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=pad_collator(tokenizer, m.max_seq_length))
    trainer.train()
    trainer.save_model(os.path.join(targs.output_dir, "final_adapter"))
    logger.info("Training complete")


if __name__ == "__main__":
    main()
