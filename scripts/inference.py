import argparse, torch
from diffusers import StableDiffusionPipeline # Hugging FaceのdiffusersライブラリからStable Diffusionのパイプラインをインポート

# メインの処理を定義する関数
def main():
    # --- コマンドライン引数の設定 ---
    # スクリプト実行時に外部から設定値を受け取るための準備
    ap = argparse.ArgumentParser()
    
    # 画像生成の土台となる、事前学習済みのStable Diffusionモデル
    ap.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    # 追加学習したLoRAの重みが保存されているディレクトリ
    ap.add_argument("--lora_dir", required=True, help="weights/sd15/lora など save_lora_weights 出力先")
    # 生成したい画像の内容を指示するテキスト（プロンプト）
    ap.add_argument("--prompt", required=True)
    # 生成される画像に含めたくない要素を指示するテキスト（ネガティブプロンプト）
    ap.add_argument("--negative", default="")
    # 画像生成のステップ数（大きいほど高精細になるが時間がかかる）
    ap.add_argument("--steps", type=int, default=30)
    # プロンプトへの忠実度（大きいほど指示に厳密に従う）
    ap.add_argument("--guidance", type=float, default=7.0)
    # 生成した画像の保存先パス
    ap.add_argument("--out", default="outputs/sample.png")
    
    # 設定した引数を解析し、argsオブジェクトに格納する
    args = ap.parse_args()

    # --- パイプラインの準備と画像生成 ---
    # ベースモデルをロードし、半精度浮動小数点(float16)でGPU(cuda)に配置する
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float16).to("cuda")
    
    # LoRAの重みをパイプラインにロードする
    pipe.load_lora_weights(args.lora_dir)
    
    # LoRAの重みをベースモデルに統合して、推論を高速化する
    pipe.fuse_lora()  # 合成して高速化（外したいときはコメントアウト）
    
    # VRAM使用量を削減するための最適化（VAEのスライス処理を有効化）
    pipe.enable_vae_slicing()
    
    # VRAM使用量を削減するための最適化（Attentionメカニズムのスライス処理を有効化）
    pipe.enable_attention_slicing("max")

    # --- 画像生成の実行と保存 ---
    # 設定したパラメータを元に、パイプラインを実行して画像を生成する
    img = pipe(args.prompt, negative_prompt=args.negative, num_inference_steps=args.steps, guidance_scale=args.guidance).images[0]
    
    # 生成された画像をファイルに保存する
    img.save(args.out)
    
    # 保存が完了したことをコンソールに表示する
    print("saved:", args.out)

# このスクリプトが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()