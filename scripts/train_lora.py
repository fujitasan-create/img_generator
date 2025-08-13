import os, yaml, argparse, random, inspect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from peft import LoraConfig

from accelerate import Accelerator
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import inspect
try:
    # diffusers 0.34.x 以降のLoRA Processor
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LoraProc
except Exception:
    # それ以前のバージョン
    from diffusers.models.attention_processor import LoRAAttnProcessor as LoraProc


# ---------- Dataset ----------
# 画像とキャプションのペアを読み込むためのPyTorch Datasetクラス
class CaptionImageDataset(Dataset):
    def __init__(self, images_dir, captions_dir, size=512):
        self.images = sorted([p for p in Path(images_dir).glob("*.jpg")])
        self.captions_dir = Path(captions_dir)
        self.size = int(size)
        assert len(self.images) > 0, "No images found"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        cap_path = self.captions_dir / (img_path.stem + ".txt")
        with open(cap_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # 画像をリサイズし、[-1, 1]の範囲に正規化してテンソルに変換
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return {"pixel_values": img_t, "caption": caption}


# ---------- Utils ----------
# 乱数シードを固定して再現性を確保する
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 設定ファイルから値を整数として安全に取得する
def _get_int(d, k, default):
    v = d.get(k, default)
    try:
        return int(v)
    except Exception:
        return int(float(v))

# 設定ファイルから値を浮動小数点数として安全に取得する
def _get_float(d, k, default):
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return float(str(v))


# UNetにLoRAレイヤーを追加する関数
def add_lora_to_unet(unet, r=8, alpha=16, dropout=0.0):
    """
    peft の LoraConfig を使って UNet に LoRA アダプタを追加。
    （この版の add_adapter は LoraConfig を受け取る）
    """
    if not hasattr(unet, "add_adapter"):
        raise RuntimeError("This diffusers/peft 組み合わせは UNet.add_adapter を実装していません。diffusers>=0.31 を推奨。")

    # LoRAを適用するターゲットモジュール（UNetのAttention層）を指定
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=target_modules,
    )

    # UNetにLoRAアダプタを追加し、有効化する
    adapter_name = "default"
    unet.add_adapter(lora_cfg, adapter_name=adapter_name)
    unet.enable_adapters()

    # 学習対象となるLoRAパラメータのみを抽出
    trainable_params = []
    for name, p in unet.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            trainable_params.append(p)
        else:
            p.requires_grad_(False)

    if len(trainable_params) == 0:
        raise RuntimeError("LoRA params が見つかりませんでした（peft/diffusers の版差の可能性）。")

    return trainable_params

# ---------- Main ----------
# 学習のメイン処理
def main(cfg):
    set_seed(int(cfg.get("seed", 42)))

    # 設定ファイルから学習パラメータを読み込み
    resolution = _get_int(cfg, "resolution", 512)
    batch_size = _get_int(cfg, "train_batch_size", 1)
    grad_acc = _get_int(cfg, "gradient_accumulation_steps", 8)
    total_steps = _get_int(cfg, "max_train_steps", 2000)
    lr = _get_float(cfg, "learning_rate", 1e-4)

    # 分散学習や混合精度学習のためのAcceleratorを初期化
    accelerator = Accelerator(gradient_accumulation_steps=grad_acc, mixed_precision="fp16")
    device = accelerator.device

    # --- Stable Diffusionの各コンポーネントをロード ---
    base_model = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # --- UNetにLoRAを追加 ---
    lora_cfg = cfg.get("lora", {}) or {}
    r = _get_int(lora_cfg, "r", 8)
    alpha = _get_int(lora_cfg, "alpha", 16)
    dropout = _get_float(lora_cfg, "dropout", 0.0)
    trainable_params = add_lora_to_unet(unet, r=r, alpha=alpha, dropout=dropout)

    # --- LoRA以外の全パラメータを凍結 ---
    for p in text_encoder.parameters(): p.requires_grad_(False)
    for p in vae.parameters(): p.requires_grad_(False)
    for p in unet.parameters(): p.requires_grad_(False)

    # ★ LoRAパラメータだけはFP32にキャストして学習可能にする（学習の安定化のため重要）
    for p in trainable_params:
        p.data = p.data.float()
        p.requires_grad_(True)


    # --- DatasetとDataLoaderの準備 ---
    ds_cfg = cfg["dataset"]
    dataset = CaptionImageDataset(ds_cfg["images_dir"], ds_cfg["captions_dir"], size=resolution)

    # バッチ処理用のcollate関数（テキストをトークン化）
    def collate(batch):
        px = torch.stack([b["pixel_values"] for b in batch])
        texts = [b["caption"] for b in batch]
        ids = tokenizer(
            texts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt",
        ).input_ids
        return {"pixel_values": px, "input_ids": ids}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)

    # --- OptimizerとSchedulerの準備 ---
    optimizer = torch.optim.AdamW([{"params": trainable_params, "lr": lr}], eps=1e-8)
    warmup = max(10, int(total_steps * 0.02))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    # --- Acceleratorでコンポーネントをラップ ---
    unet, text_encoder, vae, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, vae, optimizer, loader, lr_scheduler
    )

    # 学習中はVAEとText Encoderを評価モードに、UNetを学習モードに設定
    vae.eval()
    text_encoder.eval()
    unet.train()

    # --- 学習ループ ---
    global_step = 0
    progress = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
    while global_step < total_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                # VAEで画像を潜在空間にエンコード
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    enc_out = text_encoder(batch["input_ids"].to(device))[0]

                # 潜在変数にノイズを追加
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # UNetでノイズを予測
                model_pred = unet(noisy_latents, timesteps, enc_out).sample

                # 損失（MSE）を計算
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # 逆伝播、パラメータ更新
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process and global_step % 50 == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1
            progress.update(1)
            if global_step >= total_steps:
                break

    progress.close()

    # --- 学習済みLoRAウェイトの保存 ---
    output_dir = Path(cfg.get("output_dir", "weights/sd15/lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存のためにパイプラインを準備
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    pipe.unet = accelerator.unwrap_model(unet)

    # UNetからLoRAのパラメータ（state_dict）を抽出
    try:
        from peft import get_peft_model_state_dict
        unet_lora_state = get_peft_model_state_dict(pipe.unet, adapter_name="default")
    except Exception:
        # フォールバックとして、名前に "lora_" を含むパラメータを手動で抽出
        unet_lora_state = {k: v for k, v in pipe.unet.state_dict().items() if "lora_" in k}

    # LoRAウェイトを保存
    pipe.save_lora_weights(output_dir, unet_lora_layers=unet_lora_state)

    if accelerator.is_local_main_process:
        print(f"Saved LoRA to: {output_dir}")


if __name__ == "__main__":
    # --- スクリプトの実行 ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    # 設定ファイル(YAML)を読み込んでmain関数を実行
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
