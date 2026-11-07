"""SPMD decoder-only language model training in plain JAX, for AWS Trainium (JAX NeuronX).

The Trainium counterpart of the Paxml `LmCloudSpmd2B` example in the parent directory: a
transformer LM trained on synthetic tokens, sharded over a (data, model) device mesh with
`jax.sharding`. Parameters use Megatron-style tensor parallelism on the `model` axis; the batch
is split over the `data` axis. Only `jax` is required (no Flax / Optax).

Single node:   python train_spmd_lm.py --model-parallel 4
Multi node:    started once per node by jax.sbatch (jax.distributed.initialize()).
"""
import argparse
import math
import os
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

CONFIGS = {  # name: (layers, d_model, heads, d_ff)
    "tiny": (2, 256, 4, 1024),
    "125m": (12, 768, 12, 3072),
    "1b": (16, 2048, 16, 8192),
    "2b": (24, 2560, 20, 10240),  # close to Paxml LmCloudSpmd2B
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="125m", choices=CONFIGS)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--per-device-batch", type=int, default=4, help="sequences per data-parallel rank")
    p.add_argument("--model-parallel", type=int, default=1, help="size of the tensor-parallel `model` axis")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------- model
def param_specs(n_layers):
    """PartitionSpec per parameter: column-parallel QKV / up projections, row-parallel outputs."""
    layer = {"ln1": P(None), "wqkv": P(None, "model"), "wo": P("model", None),
             "ln2": P(None), "w_up": P(None, "model"), "w_down": P("model", None)}
    # The tied embedding is sharded on the hidden dim. Sharding it on the vocab axis gives wrong
    # gradients on JAX NeuronX (SDK 2.33); see the README.
    return {"embed": P(None, "model"), "layers": [dict(layer) for _ in range(n_layers)], "ln_f": P(None)}


def init_params(key, cfg, vocab):
    n_layers, d, _, d_ff = cfg
    keys = jax.random.split(key, 2 + 4 * n_layers)
    std = 0.02
    out_std = std / math.sqrt(2 * n_layers)

    def normal(k, shape, s):
        return (jax.random.normal(k, shape, jnp.float32) * s)

    layers = []
    for i in range(n_layers):
        k = keys[2 + 4 * i: 6 + 4 * i]
        layers.append({"ln1": jnp.ones((d,)), "wqkv": normal(k[0], (d, 3 * d), std),
                       "wo": normal(k[1], (d, d), out_std), "ln2": jnp.ones((d,)),
                       "w_up": normal(k[2], (d, d_ff), std), "w_down": normal(k[3], (d_ff, d), out_std)})
    return {"embed": normal(keys[0], (vocab, d), std), "layers": layers, "ln_f": jnp.ones((d,))}


def rms_norm(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps) * w).astype(x.dtype)


def forward(params, tokens, n_heads):
    bf = jnp.bfloat16
    b, s = tokens.shape
    x = params["embed"].astype(bf)[tokens]
    d = x.shape[-1]
    hd = d // n_heads
    mask = jnp.tril(jnp.ones((s, s), dtype=bool))
    pos = jnp.arange(s)[:, None] / (10000 ** (jnp.arange(0, hd, 2) / hd))[None, :]
    cos, sin = jnp.cos(pos).astype(bf), jnp.sin(pos).astype(bf)

    def rope(t):  # t: [b, s, h, hd]
        t1, t2 = t[..., 0::2], t[..., 1::2]
        c, sn = cos[None, :, None, :], sin[None, :, None, :]
        return jnp.stack([t1 * c - t2 * sn, t1 * sn + t2 * c], axis=-1).reshape(t.shape)

    for lp in params["layers"]:
        h = rms_norm(x, lp["ln1"])
        qkv = (h @ lp["wqkv"].astype(bf)).reshape(b, s, 3, n_heads, hd)
        q, k, v = rope(qkv[:, :, 0]), rope(qkv[:, :, 1]), qkv[:, :, 2]
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) / math.sqrt(hd)
        att = jax.nn.softmax(jnp.where(mask, att, -1e9), axis=-1).astype(bf)
        o = jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(b, s, d)
        x = x + o @ lp["wo"].astype(bf)
        h = rms_norm(x, lp["ln2"])
        x = x + jax.nn.gelu(h @ lp["w_up"].astype(bf)) @ lp["w_down"].astype(bf)
    x = rms_norm(x, params["ln_f"])
    return x @ params["embed"].astype(bf).T  # tied embeddings


def loss_fn(params, tokens, n_heads):
    logits = forward(params, tokens[:, :-1], n_heads).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    tgt = tokens[:, 1:]
    if os.environ.get("CE_IMPL", "one_hot") == "take":
        return -jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1))
    # One-hot cross-entropy: the take_along_axis backward fails with NRT_EXEC_OOB on some
    # batches on JAX NeuronX (SDK 2.33) when the logits are model-sharded; see the README.
    return -jnp.mean(jnp.sum(logp * jax.nn.one_hot(tgt, logits.shape[-1], dtype=logp.dtype), axis=-1))


# ----------------------------------------------------------------------------- optimizer (AdamW)
def adamw_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, params), "t": jnp.zeros((), jnp.int32)}


def adamw_update(params, grads, state, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1, clip=1.0):
    gnorm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, clip / (gnorm + 1e-6))
    grads = jax.tree.map(lambda g: g * scale, grads)
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    tf = t.astype(jnp.float32)
    c1, c2 = 1 - b1 ** tf, 1 - b2 ** tf

    def upd(p, m, v):
        decay = wd * p if p.ndim > 1 else 0.0
        return p - lr * ((m / c1) / (jnp.sqrt(v / c2) + eps) + decay)

    return jax.tree.map(upd, params, m, v), {"m": m, "v": v, "t": t}, gnorm


# ----------------------------------------------------------------------------- main
def main():
    a = parse()
    if os.environ.get("JAX_COORDINATOR_ADDRESS"):  # multi-process: set by jax.sbatch / run_neuron.sh
        jax.distributed.initialize(coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
                                   num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
                                   process_id=int(os.environ["JAX_PROCESS_ID"]))
    cfg = CONFIGS[a.config]
    n_layers, d, n_heads, d_ff = cfg
    devs = jax.devices()
    n = len(devs)
    assert n % a.model_parallel == 0, f"{n} devices are not divisible by --model-parallel {a.model_parallel}"
    dp = n // a.model_parallel
    mesh = Mesh(mesh_utils_devices(devs, dp, a.model_parallel), ("data", "model"))
    lead = jax.process_index() == 0
    if lead:
        print(f"backend={jax.default_backend()} processes={jax.process_count()} devices={n} "
              f"mesh(data={dp}, model={a.model_parallel}) config={a.config}", flush=True)

    specs = param_specs(n_layers)
    p_shard = jax.tree.map(lambda s: NamedSharding(mesh, s), specs, is_leaf=lambda x: isinstance(x, P))
    params = jax.jit(lambda k: init_params(k, cfg, a.vocab), out_shardings=p_shard)(jax.random.PRNGKey(a.seed))
    opt_shard = {"m": p_shard, "v": p_shard, "t": NamedSharding(mesh, P())}
    opt = jax.jit(adamw_init, out_shardings=opt_shard)(params)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    if lead:
        print(f"parameters: {n_params / 1e6:.1f} M", flush=True)

    gb = a.per_device_batch * dp
    batch_shard = NamedSharding(mesh, P("data", None))

    def lr_at(step):  # host-side schedule: the learning rate enters the step as an input
        warm = min(1.0, (step + 1) / a.warmup)
        prog = min(max((step - a.warmup) / max(1, a.steps - a.warmup), 0.0), 1.0)
        return a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    def make_batch(step):
        # Synthetic data with learnable structure: token t+1 = (a * token t + b) mod vocab for a
        # per-sequence (a, b); the first half of each sequence is random, the rest follows the rule.
        k = jax.random.fold_in(jax.random.PRNGKey(a.seed + 1), step)
        k1, k2, k3 = jax.random.split(k, 3)
        start = jax.random.randint(k1, (gb, 1), 0, a.vocab)
        mult = jax.random.randint(k2, (gb, 1), 1, 8)
        idx = jnp.arange(a.seq + 1)[None, :]
        seqs = (start + mult * idx) % a.vocab
        noise = jax.random.randint(k3, (gb, a.seq + 1), 0, a.vocab)
        return jnp.where(idx < (a.seq + 1) // 8, noise, seqs).astype(jnp.int32)

    @jax.jit
    def train_step(params, opt, tokens, lr):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, n_heads)
        params, opt, gnorm = adamw_update(params, grads, opt, lr)
        return params, opt, loss, gnorm

    make_batch = jax.jit(make_batch, out_shardings=batch_shard)
    flops_per_tok = 6 * n_params + 12 * n_layers * d * a.seq
    t_start = None
    for step in range(a.steps):
        tokens = make_batch(step)
        t0 = time.perf_counter()
        params, opt, loss, gnorm = train_step(params, opt, tokens, jnp.float32(lr_at(step)))
        loss = float(loss)
        dt = time.perf_counter() - t0
        if step == 1:
            t_start = time.perf_counter()  # steps 0-1 include compilation
        if lead and (step % a.log_every == 0 or step == a.steps - 1):
            tps = gb * a.seq / dt
            print(f"step {step:4d} loss {loss:.4f} gnorm {float(gnorm):.3f} step_time {dt * 1e3:.0f} ms "
                  f"tokens/s {tps:,.0f} TFLOP/s/device {flops_per_tok * tps / n / 1e12:.1f}", flush=True)
    if lead and t_start and a.steps > 2:
        el = time.perf_counter() - t_start
        tps = gb * a.seq * (a.steps - 2) / el
        print(f"SUMMARY steady-state tokens/s {tps:,.0f} ({tps / n:,.0f} per device) "
              f"over {a.steps - 2} steps, final loss {loss:.4f}", flush=True)
    if os.environ.get("JAX_COORDINATOR_ADDRESS"):
        jax.distributed.shutdown()


def mesh_utils_devices(devs, dp, mp):
    """Devices as a [dp, mp] grid with each model-parallel group on one process (node)."""
    import numpy as np
    by_proc = sorted(devs, key=lambda x: (x.process_index, x.id))
    return np.array(by_proc).reshape(dp, mp)


if __name__ == "__main__":
    main()
