#!/usr/bin/env python3
"""Single-GPU smoke test: drive a Megatron model as an inference engine.

This is the runtime counterpart to the static reproduction of radixark/miles#400.
It proves the `mega` env can actually run Megatron-LM's own inference stack
(`megatron.core.inference`: GPTInferenceWrapper -> TextGenerationController ->
StaticInferenceEngine) end to end on one GPU with a tiny random-weight model.

It deliberately does NOT touch miles or SGLang -- the whole point of #400 is that
Megatron can generate without SGLang. Random weights => garbage tokens; we only
assert the engine runs and returns the requested number of tokens.

Run (pick a free GPU):
    CUDA_VISIBLE_DEVICES=2 python megatron_inference_smoke.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

VOCAB = 128
HIDDEN = 64
HEADS = 4
LAYERS = 2
MAX_SEQ = 64
GEN_TOKENS = 8


def init_single_gpu_parallel() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12399")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)


def build_wrapped_model():
    from megatron.core.inference.contexts import StaticInferenceContext
    from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
        GPTInferenceWrapper,
    )
    from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
        InferenceWrapperConfig,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        use_cpu_initialization=True,
    )
    model = (
        GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            parallel_output=True,
            pre_process=True,
            post_process=True,
        )
        .cuda()
        .eval()
    )

    iwc = InferenceWrapperConfig(
        hidden_size=HIDDEN,
        inference_batch_times_seqlen_threshold=-1,
        inference_max_requests=8,
        fp32_residual_connection=False,
        params_dtype=torch.float32,
        padded_vocab_size=VOCAB,
    )
    ctx = StaticInferenceContext.from_config(iwc)
    return GPTInferenceWrapper(model, iwc, ctx)


def main() -> int:
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    init_single_gpu_parallel()
    print(f"device: {torch.cuda.get_device_name(0)} | TP=1 PP=1 initialized")

    wrapped = build_wrapped_model()
    n_params = sum(p.numel() for p in wrapped.model.parameters())
    print(f"tiny GPT built: {LAYERS} layers, hidden {HIDDEN}, vocab {VOCAB} ({n_params/1e6:.2f}M params)")

    from megatron.core.inference.engines import StaticInferenceEngine
    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )
    from megatron.training.tokenizer.tokenizer import _NullTokenizer

    class _SmokeTokenizer(_NullTokenizer):
        # The text-generation controller reads `.bos`/`.eos`; _NullTokenizer
        # raises NotImplementedError for them. Provide benign values.
        @property
        def bos(self):
            return None

        @property
        def eos(self):
            return self.eod

    tokenizer = _SmokeTokenizer(vocab_size=VOCAB)
    controller = TextGenerationController(inference_wrapped_model=wrapped, tokenizer=tokenizer)
    engine = StaticInferenceEngine(controller, max_batch_size=4, buffer_size_gb=1.0)

    sampling = SamplingParams(num_tokens_to_generate=GEN_TOKENS, temperature=1.0, top_k=1)
    prompt = "1 2 3 4 5"
    print(f"\nrunning engine.generate(prompts=[{prompt!r}], num_tokens_to_generate={GEN_TOKENS}) ...")
    results = engine.generate(prompts=[prompt], sampling_params=sampling)

    r = results[0]
    gen = getattr(r, "generated_tokens", None)
    if gen is None:
        gen = []
    elif hasattr(gen, "tolist"):
        gen = gen.tolist()
    else:
        gen = list(gen)
    print(f"  prompt_tokens   : {tokenizer.tokenize(prompt)}")
    print(f"  generated_tokens: {gen}")
    print(f"  generated_text  : {getattr(r, 'generated_text', None)!r}")

    ok = len(gen) == GEN_TOKENS
    print("\nSMOKE TEST: " + ("PASSED" if ok else f"FAILED (expected {GEN_TOKENS} tokens, got {len(gen)})"))
    print("Megatron drove autoregressive generation end-to-end as an inference engine.")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
