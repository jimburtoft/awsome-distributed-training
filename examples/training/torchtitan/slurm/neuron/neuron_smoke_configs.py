# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Single-instance smoke configs: the torchtitan-neuron Llama3 recipe settings applied to the small
"debugmodel" flavor, so the Neuron path can be checked on one trn2.3xlarge before an 8B run.

Use with: torchrun ... -m torchtitan.train --module neuron_smoke_configs --config llama3_debug_neuron_sac_eager
"""

from torchtitan.config import CompileConfig
from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan_neuron.models.llama3.config_registry import _llama3_neuron_dense_config


def llama3_debug_neuron_sac_eager():
    return _llama3_neuron_dense_config("debugmodel", llama3_debugmodel, full_ac=False)


def llama3_debug_neuron_sac_compile():
    return _llama3_neuron_dense_config(
        "debugmodel",
        llama3_debugmodel,
        full_ac=False,
        compile_config=CompileConfig(enable=True, components=["model", "loss", "grad_clip"], backend="neuron"),
    )
