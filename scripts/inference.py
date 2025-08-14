import argparse, torch
from diffusers import StableDiffusionPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora_dir", type=str, default=None, help="LoRA weights dir (optional)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--out", default="outputs/sample.png")
    ap.add_argument("--seed", type=int, default=None)         # ← 追加：A/B用
    ap.add_argument("--lora_scale", type=float, default=None) # ← 追加：任意
    args = ap.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    ).to("cuda")

    # --- LoRA は「指定がある時だけ」読み込む ---
    if args.lora_dir:
        if args.lora_scale is not None:
            pipe.load_lora_weights(args.lora_dir, weight_name=None)  # weight_name使う場合は適宜
            # 一部のバージョンでは set_adapters / set_adapters_scale が必要
            try:
                pipe.set_adapters(["default"], adapter_weights=[args.lora_scale])
            except Exception:
                pass
        else:
            pipe.load_lora_weights(args.lora_dir)

        # 読み込み済みならだけ fuse（必須ではない）
        try:
            pipe.fuse_lora()
        except Exception:
            pass

    # 省メモリ設定
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing("max")

    # 同一シードでの再現
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

    img = pipe(
        args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]

    img.save(args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()