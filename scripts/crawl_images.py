import argparse, os
from icrawler.builtin import BingImageCrawler # Webから画像を収集するicrawlerライブラリの中からBing検索用の機能をインポート

def main():
    # --- コマンドライン引数の設定 ---
    # スクリプト実行時に外部から設定値を受け取るための準備
    ap = argparse.ArgumentParser()
    # 検索キーワード（必須項目）
    ap.add_argument("--query", required=True, help="検索キーワード（例: 黒髪 ボブ 女の子 イラスト）")
    # ダウンロードする画像の最大数
    ap.add_argument("--max_num", type=int, default=300)
    # 画像の保存先ディレクトリ
    ap.add_argument("--out_dir", default="data/raw")
    # 画像の種類のフィルタ（写真、クリップアートなど）
    ap.add_argument("--type", default="photo", choices=["photo", "clipart", "line", "animatedgif"])
    # 画像の色のフィルタ（カラー、モノクロ）
    ap.add_argument("--color", default="color", choices=["color", "monochrome"])
    # 設定した引数を解析し、argsオブジェクトに格納する
    args = ap.parse_args()

    # --- 画像収集（クロール）の実行 ---
    # 保存先ディレクトリが存在しない場合は作成する
    os.makedirs(args.out_dir, exist_ok=True)
    # Bing検索の際のフィルタ条件を辞書として定義する
    filters = {"type": args.type, "color": args.color}  
    # BingImageCrawlerのインスタンスを作成し、画像の保存先（root_dir）を指定する
    crawler = BingImageCrawler(storage={"root_dir": args.out_dir})
    # 指定したキーワードとフィルタ条件で、画像の収集を開始する
    crawler.crawl(keyword=args.query, max_num=args.max_num, filters=filters)

# このスクリプトが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()