# PyTorch Native on AWS Trainium1 (trn1)

> **Not supported.** PyTorch Native (torch-neuronx 2.14, Neuron SDK 2.33) is not supported on
> Trainium1 instances (`trn1.2xlarge`, `trn1.32xlarge`, `trn1n.32xlarge`). The trn1 settings in this
> repository are included for testing only. For production training, use Trainium2 (`trn2`), which is
> fully supported and is the target of every Neuron example here.

This page collects the settings that the Neuron examples in this repository use when they are tested
on trn1. Each example's README links here instead of repeating them.

## Environment

Use the same public Neuron SDK 2.33 PyTorch Native image as on trn2:

```text
public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04
```

| Instance | Neuron devices | NeuronCores (ranks per node) | HBM per NeuronCore | EFA interfaces |
|---|---|---|---|---|
| `trn1.2xlarge` | 1 | 2 | 16 GiB | -- |
| `trn1.32xlarge` | 16 | 32 | 16 GiB | 8 (single-node testing only) |
| `trn1n.32xlarge` | 16 | 32 | 16 GiB | 16 |

Run one process per NeuronCore. `neuron-ls --json-output` reports the NeuronCore count (`nc_count` per
device) and the instance type. For multi-node tests, use `trn1n.32xlarge`.

## Recommended settings

Set these on trn1 only. On trn2, keep the defaults.

| Setting | trn1 value | trn2 value | Used by |
|---|---|---|---|
| `TORCH_NEURONX_INT64_MODE` | `packed` | `native` (default) | Distributed jobs (DDP, FSDP) |
| `NEURON_LAUNCH_BLOCKING` | `1` | unset | Jobs that combine `torch.compile` with collectives |
| Optimizer | foreach AdamW (`torch.optim.AdamW(..., fused=False)`; Hugging Face Trainer: `--optim adamw_torch`) | default | Jobs that use AdamW |
| Attention | Hugging Face models: `attn_implementation="eager"`. Direct `scaled_dot_product_attention` calls: an additive causal mask filled with the dtype's minimum, not `is_causal=True`, `-inf` or a boolean mask | default | Transformer models |
| Random initialization and dropout | create tensors on the CPU, then move them to the NeuronCore; dropout 0 | default | Models that initialize weights on the device or train with dropout |

An example README states which of these rows it needs. Rows that an example does not mention do not
apply to it.

## Examples tested on trn1

| Example | Settings | Tested on |
|---|---|---|
| [`examples/training/ddp`](../examples/training/ddp/README.md#aws-trainium) | `TORCH_NEURONX_INT64_MODE=packed` | `trn1.2xlarge`; HyperPod 2x `ml.trn1n.32xlarge` |
