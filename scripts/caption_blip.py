import argparse, os # CLIの引数をパースする/OS依存の機能を扱う（ファイルパス操作など）ライブラリ
from PIL import Image # 画像処理ライブラリ Pillow
from transformers import pipeline  # Hugging Faceの推論パイプライン
from tqdm import tqdm # ループ処理の進捗をプログレスバーで表示

# -------------------------------------------------------------------
# メイン処理
# -------------------------------------------------------------------

def main():
    # -------------------------------------------------------------------
    # 引数の定義とパース
    # -------------------------------------------------------------------
    # argparseを用いて、コマンドラインからの入力を受け付ける
    ap = argparse.ArgumentParser()
    # 入力画像が格納されているディレクトリ
    ap.add_argument("--images_dir", default="data/processed/images")
    # 生成されたキャプション（.txt）を保存するディレクトリ
    ap.add_argument("--captions_dir", default="data/processed/captions")
    # 使用するHugging Face Hub上のモデル名
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    # 一度に処理する画像の枚数（バッチサイズ）
    # VRAMの容量に応じて調整する
    ap.add_argument("--batch", type=int, default=8)
    # コマンドライン引数をパース
    args = ap.parse_args()

    # -------------------------------------------------------------------
    # セットアップ：出力ディレクトリ作成とモデルのロード
    # -------------------------------------------------------------------

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(args.captions_dir, exist_ok=True)
    # Hugging Faceのpipelineを初期化
    # "image-to-text"タスクを指定し、事前学習済みモデルをロードする
    
    cap = pipeline("image-to-text", model=args.model)

    # -------------------------------------------------------------------
    # データセットの読み込みとバッチ処理
    # -------------------------------------------------------------------

    # 指定ディレクトリから指定拡張子(.jpg, .jpeg, .png)の画像ファイルリストを取得

    imgs = [f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    # tqdmで進捗を可視化しつつ、データセットをバッチ単位でイテレーション
    for i in tqdm(range(0, len(imgs), args.batch), desc="caption"):
        # 現在のバッチに該当するファイルリストをスライス
        batch_files = imgs[i:i+args.batch]
        # ファイルパスのリストを作成
        batch_paths = [os.path.join(args.images_dir, f) for f in batch_files]
        # 各画像をPillowで開き、モデルの入力形式であるRGBに変換
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        # ★★★ モデルによる推論実行 ★★★
        # 画像バッチをパイプラインに渡し、キャプション生成を実行
        outs = cap(images)
        # -------------------------------------------------------------------
        # 結果の保存
        # -------------------------------------------------------------------
        
        # バッチ内の各画像ファイルと推論結果をzipでループ
        for f, o in zip(batch_files, outs):
            # パイプラインの出力(辞書のリスト)から生成テキストを抽出
            # .strip()で前後の空白を除去
            text = o[0]["generated_text"].strip()
            # 生成されたキャプションをUTF-8でファイルに書き込み
            with open(os.path.join(args.captions_dir, os.path.splitext(f)[0] + ".txt"), "w", encoding="utf-8") as w:
                w.write(text)

# このファイルが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()