# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Run ../train.py on AWS Trainium: GRPO with TRL 1.x in vLLM server mode.

- Generation: a vLLM on AWS Neuron server, started with the `neuron_weight_reload.NeuronWeightReload` worker
  extension and `VLLM_SERVER_DEV_MODE=1` (see train-neuron.sbatch).
- Training: GRPO on NeuronCores (PyTorch Native) with FSDP2. New policy weights reach the server through
  `trl_neuron_weight_sync` (shared directory + in-place reload) instead of NCCL.
- train.py is unchanged. Settings it hard-codes for GPUs are adjusted through the environment:
    GRPO_BATCH_SIZE        per_device_train_batch_size (train.py: 32)
    GRPO_NUM_GENERATIONS   completions per prompt (TRL default 8)
    GRPO_MAX_COMPLETION    max_completion_length (TRL default 256)
    GRPO_MAX_STEPS         stop after this many steps (default: one epoch)
    GRPO_REPORT_TO         report_to (train.py: wandb)
    GRPO_LOGGING_STEPS     logging_steps (train.py: 10)
    GRPO_ENABLE_THINKING   Qwen3 chat-template thinking mode (default 0: train.py asks for its own
                           <think>/<answer> format, and thinking mode clips most completions)
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trl_neuron_weight_sync  # noqa: E402

trl_neuron_weight_sync.install()

from trl import GRPOConfig  # noqa: E402

_post_init = GRPOConfig.__post_init__


def _neuron_post_init(self, *a, **k):
    env = os.environ.get
    if self.model_init_kwargs is None:
        self.model_init_kwargs = {"dtype": "bfloat16"}  # TRL loads the policy in float32 otherwise
    self.per_device_train_batch_size = int(env("GRPO_BATCH_SIZE", self.per_device_train_batch_size))
    self.num_generations = int(env("GRPO_NUM_GENERATIONS", self.num_generations or 8))
    self.max_completion_length = int(env("GRPO_MAX_COMPLETION", self.max_completion_length or 256))
    if env("GRPO_MAX_STEPS"):
        self.max_steps = int(env("GRPO_MAX_STEPS"))
    self.report_to = env("GRPO_REPORT_TO", "none")
    self.logging_steps = int(env("GRPO_LOGGING_STEPS", self.logging_steps))
    if self.chat_template_kwargs is None:
        self.chat_template_kwargs = {"enable_thinking": env("GRPO_ENABLE_THINKING", "0") == "1"}
    # train.py sets vllm_server_host for a separate vLLM node, which was server mode in TRL 0.x. TRL 1.x
    # defaults to colocate, so select server mode explicitly.
    self.vllm_mode = "server"
    # train.py calls torch.cuda.empty_cache() through this option; it has no Neuron equivalent.
    self.torch_empty_cache_steps = None
    # The policy logprobs are recomputed on the trainer, as in the TRL 0.x path this example was written for.
    self.vllm_importance_sampling_correction = False
    return _post_init(self, *a, **k)


GRPOConfig.__post_init__ = _neuron_post_init

if __name__ == "__main__":
    script = os.path.join(HERE, "..", "train.py")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
