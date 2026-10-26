# SPDX-License-Identifier: Apache-2.0
"""Startup admission for the real target quantization wrapper, before compile."""
import os

import torch


def prepare(model):
    if (os.environ.get("NEMO_MOE_SHARED_SUM", "1") == "0"
            or type(model).__name__ != "NemotronHForCausalLM"
            or torch.cuda.is_current_stream_capturing()):
        return
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    logger = init_logger(__name__)
    inner = model.model
    if inner.end_layer != len(inner.layers):
        return
    last = inner.layers[-1]
    skip = getattr(getattr(last, "mixer", None), "experts", None)
    admitted = 0
    for runner in model.modules():
        if not isinstance(runner, MoERunner) or runner is skip:
            continue
        if type(runner.routed_experts.quant_method).__name__ != "ModelOptNvFp4FusedMoE":
            continue
        try:
            admitted += int(runner.prepare_shared_sum_contract())
        except Exception as exc:
            # Admission is optional. No real model forward has run through
            # this helper and the runner keeps its original contract on refusal.
            logger.info("Target shared-sum startup refused layer=%s error=%s",
                        runner.layer_name, type(exc).__name__)
    logger.info("Target ModelOpt shared-sum startup admission: %d; final MoE preserved",
                admitted)
