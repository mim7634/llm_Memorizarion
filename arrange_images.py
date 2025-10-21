import os
from PIL import Image, ImageDraw, ImageFont

class ImageGridMaker:
    """
    指定されたフォルダ内の画像を読み込み、各画像の上部に余白を作成し、
    そこにファイル名を描画してからグリッド状に結合するクラス。
    """
    
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')

    def __init__(self, input_dir, max_cols, font_path, font_size=30, 
                 text_margin_height=50, base_width=None):
        """
        初期化メソッド。

        Args:
            input_dir (str): 画像ファイルが格納されているフォルダのパス。
            max_cols (int): 横に並べる画像の最大枚数（列数）。
            font_path (str): テキスト描画に使用するTrueTypeフォントファイル（.ttf, .ttc）のパス。
            font_size (int): 描画するテキストのフォントサイズ。
            text_margin_height (int): ファイル名を描画するために画像の上部に追加する余白の高さ（ピクセル）。
            base_width (int, optional): グリッド内の全ての画像の幅を揃えるための基準幅。
        """
        self.input_dir = input_dir
        self.max_cols = max_cols
        self.base_width = base_width
        self.text_margin_height = text_margin_height # 👈 余白の高さを保持
        self.images_with_names = []
        
        # フォントオブジェクトの準備
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"エラー: フォントファイルが見つからないか、読み込めません: {font_path}。デフォルトフォントを使用します。")
            self.font = ImageFont.load_default() 
        except Exception as e:
            print(f"フォントの読み込み中に予期せぬエラー: {e}。デフォルトフォントを使用します。")
            self.font = ImageFont.load_default() 

    def _load_images(self):
        # (変更なし: 画像オブジェクトとファイル名を保持)
        file_names = sorted([
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(self.SUPPORTED_FORMATS)
        ])
        if not file_names:
            print(f"エラー: フォルダ '{self.input_dir}' に画像ファイルが見つかりませんでした。")
            return False
        try:
            for file_name in file_names:
                path = os.path.join(self.input_dir, file_name)
                img = Image.open(path).convert("RGB")
                img.load()
                self.images_with_names.append((img, file_name))
            return True
        except Exception as e:
            print(f"エラー: 画像の読み込み中に問題が発生しました - {e}")
            return False

    def _resize_and_add_margin(self):
        """画像を基準幅にリサイズし、上部にテキスト描画用の余白を追加します。"""
        
        if not self.images_with_names:
            return

        # 基準幅を決定
        if self.base_width is None:
            self.base_width = self.images_with_names[0][0].width 
        
        margined_list = []
        for img, file_name in self.images_with_names:
            
            # 1. 基準幅に合わせてリサイズ (画像本体の縦横比は維持)
            if img.width != self.base_width:
                aspect_ratio = img.height / img.width
                new_height = int(self.base_width * aspect_ratio)
                img = img.resize((self.base_width, new_height))

            original_height = img.height
            
            # 2. 余白（マージン）を追加した新しい画像を作成
            # 新しい高さ = 元の高さ + 余白の高さ
            new_total_height = original_height + self.text_margin_height
            
            # 余白部分を白にした新しい画像を作成
            margined_img = Image.new('RGB', (self.base_width, new_total_height), color='white')
            
            # 3. 元の画像を余白の下（Y座標=余白の高さ）に貼り付け
            # 余白の高さ分をYオフセットとする
            margined_img.paste(img, (0, self.text_margin_height)) 

            # 4. ファイル名を描画 (この時点で描画してしまう)
            margined_img_with_text = self._draw_text_on_image(margined_img, file_name)
            
            # 余白付き画像とファイル名を保持 (ファイル名は描画済みのため、ここでは余白付き画像のみを更新)
            margined_list.append((margined_img_with_text, file_name))
        
        self.images_with_names = margined_list

    def _draw_text_on_image(self, img, text, padding=5, text_color="black"):
        """画像の上部マージンエリアにテキストを描画します。"""
        
        draw = ImageDraw.Draw(img)
        
        # テキストの描画位置 (X座標: 左端からpadding, Y座標: 上端からpadding)
        # 余白エリア (0 ~ self.text_margin_height) に描画
        position = (padding, padding) 
        
        draw.text(position, text, fill=text_color, font=self.font)
        
        return img

    def create_grid(self, output_filename="combined_grid_with_margin.jpg"):
        """画像をグリッド状に結合し、ファイル名を描画して保存します。"""
        
        if not self._load_images():
            return
        
        # リサイズと余白追加、およびテキスト描画を実行
        self._resize_and_add_margin()

        num_images = len(self.images_with_names)
        num_rows = (num_images + self.max_cols - 1) // self.max_cols
        
        # 各行の高さを決定 (この時点ではすでに余白込みの高さになっている)
        row_heights = []
        for i in range(num_rows):
            start_index = i * self.max_cols
            end_index = min((i + 1) * self.max_cols, num_images)
            # 画像オブジェクトのリストを取得 (すでに余白込みの高さ)
            row_images = [item[0] for item in self.images_with_names[start_index:end_index]]
            if not row_images: continue
            row_height = max(img.height for img in row_images)
            row_heights.append(row_height)

        # 全体の幅と高さを計算
        grid_width = self.base_width * self.max_cols
        grid_height = sum(row_heights)
        
        grid_img = Image.new('RGB', (grid_width, grid_height), color='white')

        # 画像をグリッドに配置
        y_offset = 0
        for i in range(num_rows):
            start_index = i * self.max_cols
            end_index = min((i + 1) * self.max_cols, num_images)
            if i >= len(row_heights): break # 念のためインデックスチェック
            current_row_height = row_heights[i]
            x_offset = 0
            
            for img_tuple in self.images_with_names[start_index:end_index]:
                img, _ = img_tuple
                
                # 余白付き画像（テキスト描画済み）をグリッドに貼り付け
                grid_img.paste(img, (x_offset, y_offset))
                x_offset += self.base_width 

            y_offset += current_row_height

        # 結果を保存
        grid_img.save(output_filename)
        print(f"✅ 画像の結合とファイル名の描画が完了しました。ファイル名: {output_filename}")


# --- 実行例（ユーザー設定を反映） ---
if __name__ == "__main__":
    
    # 1. 画像フォルダのパス
    IMAGE_DIR = "analyze"

    # 2. 横に並べる最大枚数
    COLUMNS = 4 

    # 3. 出力ファイル名
    OUTPUT = "analyze/arrange_epc_memorization_data_margin.png" # PNG形式で保存する場合

    # 4. フォントの設定 (環境に合わせて修正が必要)
    FONT_PATH = "arial.ttf" # 👈 実行環境に合わせて修正してください
    FONT_SIZE = 40
    
    # 5. テキスト描画用の上部余白の高さ (文字が余裕をもって入るように調整)
    TEXT_MARGIN = 50 
    
    # フォルダ 'analyze' が存在しない場合は作成
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print(f"注意: フォルダ '{IMAGE_DIR}' が見つからなかったため作成しました。画像をここに入れてください。")
    
    # クラスのインスタンス化と実行
    maker = ImageGridMaker(
        input_dir=IMAGE_DIR, 
        max_cols=COLUMNS, 
        font_path=FONT_PATH, 
        font_size=FONT_SIZE,
        text_margin_height=TEXT_MARGIN # 👈 余白の高さを渡す
    )
    maker.create_grid(output_filename=OUTPUT)