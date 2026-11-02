"""LoRA parameter arithmetic from Lecture 4.

    python finetune/lora_math.py                  # the two worked examples from the slides
    python finetune/lora_math.py 896 896 8        # any d_out d_in r (896 = Qwen2.5-0.5B hidden size)
"""

import sys


def lora_params(d_out: int, d_in: int, r: int) -> tuple[int, int, float]:
    full = d_out * d_in                 # full update dW: d_out x d_in
    lora = r * (d_in + d_out)           # A: r x d_in, B: d_out x r
    return full, lora, full / lora


def show(d_out, d_in, r):
    full, lora, ratio = lora_params(d_out, d_in, r)
    print(f"W in R^{d_out}x{d_in}, r={r}: full={full:,}  LoRA={lora:,}  -> {ratio:.0f}x fewer trainable params "
          f"(for this one matrix, not the whole model)")


def weight_storage_gb(n_params: float):
    for name, bytes_ in (("FP32", 4), ("FP16", 2), ("8-bit", 1), ("4-bit", 0.5)):
        print(f"  {name:6} {n_params * bytes_ / 1e9:6.2f} GB  (raw weights only; training needs more)")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        show(*map(int, sys.argv[1:]))
    else:
        show(4096, 4096, 8)   # slide: 256x
        show(2048, 2048, 8)   # slide: 128x
        show(896, 896, 8)     # Qwen2.5-0.5B q_proj
        print("Raw weight storage for a 7B model:")
        weight_storage_gb(7e9)
        print("Raw weight storage for Qwen2.5-0.5B (~494M params):")
        weight_storage_gb(494e6)
