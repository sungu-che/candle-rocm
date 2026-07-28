---
title: Z-Image-Ultra-Fast
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
suggested_hardware: zero-a10g
suggested_storage: large
preload_from_hub:
  - Tongyi-MAI/Z-Image-Turbo
  - Tongyi-MAI/Z-Image
---

# ⚡ Z-Image-Ultra-Fast

Dual-mode AI image generation powered by **Z-Image** from Tongyi-MAI.

## Modes

| Mode | Model | Steps | Speed |
|------|-------|:-----:|-------|
| **Turbo** | Z-Image-Turbo | 8–9 | ~1–2s |
| **Mean Cache** | Z-Image + JVP | 13–25 | ~3–8s |

- **Turbo**: Unconditioned (guidance_scale=0), designed for ultra-fast inference.
- **Mean Cache**: JVP-accelerated multi-step with adaptive step skipping for quality+speed.

## Quick Start

1. Pick **Turbo (Fast)** or **Z-Image + Mean Cache (Quality)**
2. Enter your prompt
3. Adjust steps via the ⚡ Steps slider
4. Click **🚀 Generate Image**

## Model Credits

- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — Apache 2.0
- [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) — Apache 2.0
- Mean Cache by [@multimodalart](https://huggingface.co/multimodalart)
- Original demo by [@mrfakename](https://x.com/realmrfakename)