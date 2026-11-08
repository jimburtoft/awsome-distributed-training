# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""TRL server-mode weight sync to vLLM on AWS Neuron.

TRL's `VLLMClient` pushes new policy weights to the vLLM server over vLLM's NCCL weight-transfer engine, which
needs CUDA. `NeuronVLLMClient` keeps TRL's generation path unchanged and replaces only the weight push:

- `update_named_params()` writes the (already PEFT-merged, FSDP-gathered) tensors that TRL streams to a
  safetensors checkpoint on storage shared with the server, then
- calls the server's `/collective_rpc` endpoint with `reload_weights_from`, provided by the
  `neuron_weight_reload.NeuronWeightReload` worker extension, which copies the weights into the running model
  in place (no recompilation).

Server: `VLLM_SERVER_DEV_MODE=1 vllm serve <model> --worker-extension-cls neuron_weight_reload.NeuronWeightReload ...`
Trainer: call `install()` before constructing `GRPOTrainer`, and set `NEURON_WEIGHT_SYNC_DIR` to the shared path.
"""

import json
import os
import shutil
import time

import torch
from safetensors.torch import save_file

from trl.generation import vllm_client as _vllm_client
from trl.generation import vllm_generation as _vllm_generation


class NeuronVLLMClient(_vllm_client.VLLMClient):
    sync_dir = os.environ.get("NEURON_WEIGHT_SYNC_DIR", "/fsx/weight_sync")
    # The server reads the files from its own view of the shared directory.
    server_sync_dir = os.environ.get("NEURON_WEIGHT_SYNC_SERVER_DIR", sync_dir)

    def init_communicator(self, device=0):
        # No NCCL group: weights travel through shared storage. Copy the model's config and tokenizer files once,
        # so that each sync directory is a loadable Hugging Face checkpoint.
        os.makedirs(self.sync_dir, exist_ok=True)
        self.sync_count = 0
        self.last_sync = {}

    def close_communicator(self):
        pass

    def update_named_params(self, metadata, named_params):
        t0 = time.perf_counter()
        # FSDP2 with bf16 mixed precision gathers the fp32 master weights; the server runs in bf16, so send bf16
        # (half the bytes to write and read).
        dtype = getattr(torch, os.environ.get("NEURON_WEIGHT_SYNC_DTYPE", "bfloat16"))
        tensors = {name: t.detach().to("cpu", dtype=dtype if t.is_floating_point() else t.dtype, copy=True).contiguous()
                   for name, t in named_params}
        # Tied-embedding models (e.g. Qwen3-0.6B/1.7B/4B) stream no lm_head; vLLM on AWS Neuron 0.29.0.1.2.0 still
        # expects one (vllm-project/vllm-neuron#50), so write the tied tensor explicitly.
        if "lm_head.weight" not in tensors and "model.embed_tokens.weight" in tensors:
            tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
        t_gather = time.perf_counter()
        os.makedirs(self.sync_dir, exist_ok=True)
        tmp = os.path.join(self.sync_dir, "model.safetensors.tmp")
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, os.path.join(self.sync_dir, "model.safetensors"))
        t_save = time.perf_counter()
        resp = self._post(f"{self.base_url}/collective_rpc",
                          json={"method": "reload_weights_from", "args": [self.server_sync_dir], "timeout": 600})
        t_reload = time.perf_counter()
        self.sync_count += 1
        self.last_sync = {"tensors": len(tensors), "gather_s": round(t_gather - t0, 2),
                          "save_s": round(t_save - t_gather, 2), "reload_s": round(t_reload - t_save, 2),
                          "server": resp.get("results", [None])[0]}
        print(f"[neuron-weight-sync] #{self.sync_count} {json.dumps(self.last_sync)}", flush=True)

    def reset_prefix_cache(self):
        try:
            super().reset_prefix_cache()
        except Exception:  # endpoint is only present in dev mode; a stale prefix cache is harmless here
            pass


def prepare_sync_dir(model_dir: str, sync_dir: str | None = None):
    """Copy config, tokenizer and generation files of the served model into the sync directory."""
    sync_dir = sync_dir or NeuronVLLMClient.sync_dir
    os.makedirs(sync_dir, exist_ok=True)
    for f in os.listdir(model_dir):
        if f.endswith(".json") or f.endswith(".jinja") or f.endswith(".txt") or f.endswith(".model"):
            if f != "model.safetensors.index.json":
                shutil.copy(os.path.join(model_dir, f), os.path.join(sync_dir, f))


def install():
    """Make TRL's server mode use NeuronVLLMClient.

    The trainer image (torch-neuronx) does not include vLLM, which server mode only needs on the server. TRL checks
    that the vllm package is importable before it builds the client, so report it as available for that check.
    """
    _vllm_generation.VLLMClient = NeuronVLLMClient
    _vllm_generation.is_vllm_available = lambda *a, **k: True
    _vllm_client.is_vllm_available = lambda *a, **k: True
    return NeuronVLLMClient
