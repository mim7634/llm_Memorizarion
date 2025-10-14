# 1_prepare_data.py - dataフォルダにファイルを保存するように修正

import random
import string
import os

# --- 設定: dataフォルダのパス ---
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.txt")
EVAL_PATH = os.path.join(DATA_DIR, "eval_data.txt")

def generate_random_string(length=50):
    """指定された長さのランダムな英数字文字列を生成する"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def create_dataset(num_samples=100):
    """dataフォルダを作成し、訓練データと検証データを保存する"""
    
    # dataフォルダが存在しない場合は作成
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # データ生成と分割 (訓練90件、検証10件)
    data = [generate_random_string() for _ in range(num_samples)]
    train_data = data[:90]
    eval_data = data[90:] 

    # 訓練データをファイルに保存
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line + "\n")
            
    # 検証データを別のファイルに保存
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for line in eval_data:
            f.write(line + "\n")

    print(f"訓練データ: {len(train_data)}件を '{TRAIN_PATH}' に保存しました。")
    print(f"検証データ: {len(eval_data)}件を '{EVAL_PATH}' に保存しました。")

if __name__ == "__main__":
    create_dataset()