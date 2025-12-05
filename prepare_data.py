import random
import string
import os
import hashlib

class PREPARE_DATA:
    """
    データ生成とファイル保存を行うクラス。
    訓練データと検証データの件数はインスタンス初期化時に設定される。
    """
    def __init__(self, folder_name="data", random_string_length=34, hash_suffix_length=12, train_data_num=10000, eval_data_num=1000):
        # フォルダとパスの設定
        self.DATA_DIR = folder_name
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")
        self.EVAL_PATH = os.path.join(self.DATA_DIR, "eval_data.txt")

        # --- データ形式の定義 ---
        # ★修正: データ部分を32桁にする
        self.RANDOM_STRING_LENGTH = 32 
        # ★修正: hash_suffix_length は 12 で固定
        self.HASH_SUFFIX_LENGTH = hash_suffix_length
        self.TRAIN_DATA_NUM = train_data_num
        self.EVAL_DATA_NUM = eval_data_num
        
        # ★修正: IDの桁数を5桁に固定 (最大20000件に対応)
        self.INDEX_WIDTH = 5 

    def generate_random_string(self):
        """インスタンス変数で指定された長さのランダムな英数字文字列を生成する"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(self.RANDOM_STRING_LENGTH))

    def create_dataset(self):
        """
        dataフォルダを作成し、設定された件数に基づいて訓練データと検証データを保存する。
        """
        
        # dataフォルダが存在しない場合は作成 (再帰的にディレクトリを作成)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
        total_num = self.TRAIN_DATA_NUM + self.EVAL_DATA_NUM
        
        print(f"ディレクトリ '{self.DATA_DIR}' に合計 {total_num} 件のデータを生成します...")

        # データ生成と分割
        # 必要な総件数分のランダム文字列プールを作成
        rand_data_pool = [self.generate_random_string() for _ in range(total_num)]
        
        # インデックスフォーマットの定義 (5桁なので "05d")
        index_format = f"0{self.INDEX_WIDTH}d"

        # --- 訓練データ生成 ---
        train_data = []
        # インデックス i は 0 から TRAIN_DATA_NUM - 1 まで
        for i in range(self.TRAIN_DATA_NUM):
            data_str = rand_data_pool[i]
            # インスタンス変数 self.HASH_SUFFIX_LENGTH を使用
            hash_suffix = hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:self.HASH_SUFFIX_LENGTH]
            # 設定された桁数でインデックスをフォーマット
            combined_string = f"{i:{index_format}}:{data_str}{hash_suffix}"
            train_data.append(combined_string)

        # --- 検証データ生成 ---
        eval_data = []
        # インデックス i は TRAIN_DATA_NUM から total_num - 1 まで
        for i in range(self.TRAIN_DATA_NUM, total_num):
            data_str = rand_data_pool[i]
            # インスタンス変数 self.HASH_SUFFIX_LENGTH を使用
            hash_suffix = hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:self.HASH_SUFFIX_LENGTH]
            # 設定された桁数でインデックスをフォーマット
            combined_string = f"{i:{index_format}}:{data_str}{hash_suffix}"
            eval_data.append(combined_string)


        # 訓練データをファイルに保存
        with open(self.TRAIN_PATH, "w", encoding="utf-8") as f:
            for line in train_data:
                f.write(line + "\n")
                
        # 検証データを別のファイルに保存
        with open(self.EVAL_PATH, "w", encoding="utf-8") as f:
            for line in eval_data:
                f.write(line + "\n")

        print(f"完了: 訓練データ {len(train_data)}件, 検証データ {len(eval_data)}件. 形式: {self.INDEX_WIDTH}桁ID + 1桁':' + {self.RANDOM_STRING_LENGTH}桁データ + {self.HASH_SUFFIX_LENGTH}桁ハッシュ = 50文字")

if __name__ == "__main__":
    # --- 変則ステップのリスト定義 ---
    # 10〜90 (10刻み), 100〜900 (100刻み), 1000〜2750 (250刻み), 3000〜20000 (500刻み)
    data_size_list = (
        list(range(10, 100, 10)) +
        list(range(100, 1000, 100)) +
        list(range(1000, 3000, 250)) +
        list(range(3000, 20001, 500))
    )
    # 重複削除とソート（念のため）
    data_size_list = sorted(list(set(data_size_list)))

    for i in data_size_list:
        # 検証データが最低1件はあるように調整 (iの10%または1件)
        eval_num = int(i * 0.1)
        if eval_num < 1: eval_num = 1
        
        prepare_data = PREPARE_DATA(
            folder_name="data/"+str(i),  # ディレクトリパス (自動作成されます)
            # random_string_length=34 は PREPARE_DATA.__init__ で 32 に上書きされる
            hash_suffix_length=12, 
            train_data_num=i, 
            eval_data_num=eval_num
        )
        prepare_data.create_dataset()