# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Run ../src/train_wordle_grpo.py on AWS Trainium with TRL 1.x in vLLM server mode.

- Generation: a vLLM on AWS Neuron server (`--vllm-mode server --vllm-server-url ...`), started with the
  `neuron_weight_reload.NeuronWeightReload` worker extension and `VLLM_SERVER_DEV_MODE=1`.
- Training: GRPO on NeuronCores (PyTorch Native). New policy weights reach the server through
  `trl_neuron_weight_sync` (shared directory + in-place reload) instead of NCCL.
- TRL 1.x no longer ships `trl.experimental.openenv.generate_rollout_completions` (colocate-only in TRL 0.26).
  An equivalent for server mode is registered under that name, so the training script runs unchanged.
"""

import os
import runpy
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trl_neuron_weight_sync  # noqa: E402

trl_neuron_weight_sync.install()

from trl import GRPOConfig  # noqa: E402


_rank_clients = {}


def _client_for(trainer):
    """The main process owns TRL's client (it also syncs weights). Other ranks only generate, so each gets a
    plain HTTP client of its own: multi-turn episodes take a different number of turns on each rank, so the
    generation calls cannot be collectives."""
    gen = trainer.vllm_generation
    if getattr(gen, "vllm_client", None) is not None:
        return gen.vllm_client
    key = id(trainer)
    if key not in _rank_clients:
        from trl.generation.vllm_client import VLLMClient
        base_url = gen.server_base_url or f"http://{gen.server_host}:{gen.server_port}"
        _rank_clients[key] = VLLMClient(base_url=base_url, connection_timeout=gen.server_timeout)
    return _rank_clients[key]


def generate_rollout_completions(trainer, prompts, *, generation_overrides=None, as_chat=None):
    """Generate one completion per prompt from the vLLM server, with token ids and per-token logprobs."""
    if not prompts:
        return []
    kwargs = dict(n=1, temperature=trainer.temperature, max_tokens=trainer.max_completion_length, logprobs=0,
                  top_k=0 if trainer.top_k is None else trainer.top_k,
                  top_p=1.0 if trainer.top_p is None else trainer.top_p)
    kwargs.update(generation_overrides or {})
    out = _client_for(trainer).generate(list(prompts), **kwargs)
    tok = trainer.processing_class
    return [{"prompt_ids": p_ids, "completion_ids": c_ids,
             "logprobs": [lp[0] if isinstance(lp, list) else lp for lp in lps],
             "text": tok.decode(c_ids, skip_special_tokens=True)}
            for p_ids, c_ids, lps in zip(out["prompt_ids"], out["completion_ids"], out["logprobs"])]


openenv = types.ModuleType("trl.experimental.openenv")
openenv.generate_rollout_completions = generate_rollout_completions
sys.modules["trl.experimental.openenv"] = openenv

# The environment server closes idle WebSocket sessions. A weight sync can take longer than that, so reconnect
# and retry once when the session was closed.
def _install_env_reconnect():
    from openenv.core.env_client import EnvClient
    from websockets.exceptions import ConnectionClosed

    orig = EnvClient._send_and_receive

    def _send_and_receive(self, message):
        try:
            return orig(self, message)
        except ConnectionClosed:
            self._ws = None
            return orig(self, message)

    EnvClient._send_and_receive = _send_and_receive


_install_env_reconnect()

# Settings the training script does not expose: load the model in bf16 (the default is float32), etc.
_post_init = GRPOConfig.__post_init__


def _bf16_post_init(self, *a, **k):
    if self.model_init_kwargs is None:
        self.model_init_kwargs = {"dtype": "bfloat16"}
    self.bf16 = True
    # The training script passes top_p=None (its --top-p default); TRL 1.x expects a float.
    if self.top_p is None:
        self.top_p = 1.0
    # Full-parameter training of the 1.7B policy needs activation checkpointing to fit a NeuronCore.
    self.gradient_checkpointing = True
    # The TRL 0.26 path computed the policy logprobs on the trainer; keep that (no importance-sampling correction).
    self.vllm_importance_sampling_correction = False
    return _post_init(self, *a, **k)


GRPOConfig.__post_init__ = _bf16_post_init

if __name__ == "__main__":
    script = os.path.join(HERE, "..", "src", "train_wordle_grpo.py")
    args = sys.argv[1:]
    # Each environment server accepts one session. WORDLE_ENV_URLS="url0,url1,..." gives every rank its own
    # (on Kubernetes, one Service per replica, or one Service whose replicas >= the number of ranks).
    urls = [u for u in os.environ.get("WORDLE_ENV_URLS", "").split(",") if u]
    if urls:
        url = urls[int(os.environ.get("LOCAL_RANK", "0")) % len(urls)]
        if "--env-url" in args:
            args[args.index("--env-url") + 1] = url
        else:
            args += ["--env-url", url]
    sys.argv = [script] + args
    runpy.run_path(script, run_name="__main__")
