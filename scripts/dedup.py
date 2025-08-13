import argparse, os, shutil
from PIL import Image # 画像処理ライブラリPillow
import imagehash # 画像の類似度を判定するためのハッシュ値を計算するライブラリ
from tqdm import tqdm # ループ処理の進捗をプログレスバーで表示

def phash(img_path):
    """
    画像ファイルのパスを受け取り、Perceptual Hash (pHash)を計算して返すヘルパー関数。
    pHashは、画像の内容に基づいたハッシュ値で、類似画像の判定に用いる。
    """
    try:
        # 画像ファイルを開き、形式を標準的なRGBに統一する
        img = Image.open(img_path).convert("RGB")
        # pHashを計算して返す
        return imagehash.phash(img)
    except Exception:
        # 破損したファイルなどでエラーが起きた場合はNoneを返す
        return None

def main():
    # --- コマンドライン引数の設定 ---
    ap = argparse.ArgumentParser()
    
    # 処理対象の画像が含まれるソースディレクトリ
    ap.add_argument("--src", default="data/raw")
    # 重複除去後の画像を保存するディレクトリ
    ap.add_argument("--dst", default="data/interim")
    # 類似画像と判定するpHashの差（ハミング距離）のしきい値
    ap.add_argument("--threshold", type=int, default=8, help="ハミング距離の許容（小さいほど厳しめ）")
    
    # 設定した引数を解析し、argsオブジェクトに格納する
    args = ap.parse_args()

    # --- 画像の重複除去処理 ---
    # 保存先ディレクトリが存在しない場合は作成する
    os.makedirs(args.dst, exist_ok=True)
    
    # 保持する画像のハッシュ値を格納するリスト
    hashes = []
    # 保持した画像の枚数をカウントする変数
    kept = 0
    
    # os.walkを使って、ソースディレクトリ内の全てのファイルを再帰的に走査する
    for root, _, files in os.walk(args.src):
        # tqdmを使って、ファイル処理の進捗をプログレスバーで表示する
        for f in tqdm(files, desc="dedup"):
            # ファイルのフルパスを取得
            src_path = os.path.join(root, f)
            
            # 指定された画像拡張子でなければ処理をスキップ
            if not f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp",".gif")):
                continue
            
            # ヘルパー関数を使って画像のpHashを計算
            h = phash(src_path)
            
            # ハッシュが計算できなかった場合（ファイル破損など）はスキップ
            if h is None: 
                continue
            
            # --- ここからが重複判定の核となる部分 ---
            # 現在の画像のハッシュ(h)と、既に保持している画像のハッシュリスト(hashes)内の各ハッシュ(h2)との
            # ハミング距離（h - h2）を計算し、一つでもしきい値以下になるものがあれば重複とみなす
            if any(h - h2 <= args.threshold for h2 in hashes):
                # 重複しているため、この画像はスキップする
                continue
            
            # 重複していない場合、ハッシュをリストに追加
            hashes.append(h)
            
            # shutil.copy2でファイルをメタデータごとコピーする
            shutil.copy2(src_path, os.path.join(args.dst, f))
            
            # 保持した画像のカウンタを増やす
            kept += 1
            
    # 最終的に保持した画像の枚数を表示
    print(f"kept: {kept}")

# このスクリプトが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()