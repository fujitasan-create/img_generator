import argparse, os
from PIL import Image # Pythonの画像処理ライブラリPillow
from tqdm import tqdm # ループ処理の進捗をプログレスバーで表示

# メインの処理を定義する関数
def main():
    # --- コマンドライン引数の設定 ---
    ap = argparse.ArgumentParser()
    
    # 処理対象の画像が含まれるソースディレクトリ
    ap.add_argument("--src", default="data/interim")
    # 処理後の画像を保存するディレクトリ
    ap.add_argument("--dst", default="data/processed/images")
    # リサイズ後の目標サイズ（正方形の一辺の長さ）
    ap.add_argument("--size", type=int, default=512, help="SD1.5=512, SDXL=768")
    
    # 設定した引数を解析し、argsオブジェクトに格納する
    args = ap.parse_args()

    # --- 画像のリサイズとセンタリング処理 ---
    # 保存先ディレクトリが存在しない場合は作成する
    os.makedirs(args.dst, exist_ok=True)
    
    # os.walkを使って、ソースディレクトリ内の全てのファイルとサブディレクトリを再帰的に走査する
    for root, _, files in os.walk(args.src):
        # tqdmを使って、ファイル処理の進捗をプログレスバーで表示する
        for f in tqdm(files, desc="resize"):
            # ファイルの拡張子が指定された画像形式でなければ、処理をスキップする
            if not f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")): 
                continue
            
            # ファイルのフルパスを取得
            p = os.path.join(root, f)
            
            # 画像処理中にエラーが発生してもプログラムが停止しないようにtry-exceptで囲む
            try:
                # 画像を開き、RGBAやグレースケールなどを標準的なRGB形式に統一する
                img = Image.open(p).convert("RGB")
                
                # 画像のアスペクト比を維持したまま、指定したサイズ内に収まるように縮小する
                # Image.LANCZOSは高品質なリサイズアルゴリズム
                img.thumbnail((args.size, args.size), Image.LANCZOS)
                
                # 指定したサイズの正方形の白い背景（キャンバス）を作成する
                canvas = Image.new("RGB", (args.size, args.size), (255, 255, 255))
                
                # 縮小した画像をキャンバスの中央に配置するための座標を計算する
                x = (args.size - img.width) // 2
                y = (args.size - img.height) // 2
                
                # 計算した座標に画像を貼り付ける（背景と合成する）
                canvas.paste(img, (x, y))
                
                # 最終的な画像をJPEG形式、品質95で保存する。ファイル名は元の名前を引き継ぐ。
                canvas.save(os.path.join(args.dst, os.path.splitext(f)[0] + ".jpg"), quality=95)
            except Exception:
                # 途中で壊れた画像など、何らかのエラーが起きても無視して次のファイルへ進む
                pass

# このスクリプトが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()