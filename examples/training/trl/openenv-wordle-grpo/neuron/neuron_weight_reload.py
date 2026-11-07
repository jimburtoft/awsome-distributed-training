# vLLM worker extension (vllm-neuron 0.29): reload model weights in place from an HF-format checkpoint dir.
#
# The model's load_weights() uses load_state_dict(assign=True), which swaps in new Parameter objects that
# lose the per-parameter weight_loader (TP shard / QKV fusion transform), so it cannot be called twice.
# Instead, read and shard the checkpoint with the same loader the model uses, then copy_() into the
# existing device tensors. The compiled graphs take the weights as inputs, so nothing is recompiled.
import time, torch


class NeuronWeightReload:
    def _mappings(self, model, path):
        # Capture the mappings that the model's own load_weights() would build for this checkpoint.
        import vllm_neuron.utils.checkpoints as ck
        captured = {}
        orig = ck.SafetensorsCheckpoint.load_sharded_pipelined

        def spy(self_ck, rank, world, m, mappings, device, *a, **k):
            captured.update(mappings)
            raise _Captured()

        ck.SafetensorsCheckpoint.load_sharded_pipelined = spy
        try:
            model.load_weights(path, torch.device("cpu"), None)
        except _Captured:
            pass
        finally:
            ck.SafetensorsCheckpoint.load_sharded_pipelined = orig
        return captured

    def reload_weights_from(self, path: str):
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
        runner = self.model_runner
        model = runner.model
        t0 = time.perf_counter()
        # A fresh meta-device copy of the model still has every parameter's weight_loader attached.
        if not hasattr(self, "_reload_template"):
            from vllm import ModelRegistry
            cfg = runner.vllm_config.model_config
            cls, _ = ModelRegistry.resolve_model_cls(cfg.architecture, model_config=cfg)
            from vllm.config import set_current_vllm_config
            with set_current_vllm_config(runner.vllm_config), torch.device("meta"):
                self._reload_template = cls.from_configs(cfg.hf_config, runner.neuron_config)
        template = self._reload_template
        mappings = self._mappings(template, path)
        tp = getattr(model, "world_size", 1)
        rank = getattr(model, "rank", 0)
        result = SafetensorsCheckpoint(path, None).load_sharded_pipelined(rank, tp, template, mappings, torch.device("cpu"))
        params = dict(model.named_parameters())
        n = 0
        with torch.no_grad():
            for name, cpu_t in result.state_dict.items():
                p = params.get(name)
                if p is None:
                    continue
                p.copy_(cpu_t.to(dtype=p.dtype).to(p.device))
                n += 1
        return {"seconds": round(time.perf_counter() - t0, 2), "copied": n, "params": len(params)}

    def param_checksum(self, name_substr: str = "layers.0.self_attn"):
        for n, p in self.model_runner.model.named_parameters():
            if name_substr in n:
                return n, float(p.detach().cpu().float().sum())
        return None


class _Captured(Exception):
    pass
