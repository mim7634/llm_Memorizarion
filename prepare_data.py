# 1_prepare_data.py - 最終修正版

import random
import string
import os
import hashlib

# --- 設定: dataフォルダのパス ---
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.txt")
EVAL_PATH = os.path.join(DATA_DIR, "eval_data.txt")

# --- データ形式の定義 ---
RANDOM_STRING_LENGTH = 34  # ランダム文字列の長さ
HASH_SUFFIX_LENGTH = 12    # ハッシュ値の冒頭12桁
TRAIN_DATA_NUM = 1000
EVAL_DATA_NUM = 10

def generate_random_string(length=RANDOM_STRING_LENGTH): # ★修正: length=34の後のコロンを削除
    """指定された長さのランダムな英数字文字列を生成する"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def create_dataset(num_samples=110):
    """dataフォルダを作成し、訓練データと検証データを保存する"""
    
    # dataフォルダが存在しない場合は作成
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # データ生成と分割 (訓練100件、検証10件)
    rand_data_pool = [generate_random_string() for _ in range(num_samples)]
    
    # 訓練データ (00:ランダム34桁ハッシュ12桁)
    train_data = []
    for i in range(TRAIN_DATA_NUM):
        data_str = rand_data_pool[i]
        hash_suffix = hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:HASH_SUFFIX_LENGTH]
        combined_string = f"{i:02d}:{data_str}{hash_suffix}"
        train_data.append(combined_string)

    # 検証データ (100:ランダム34桁ハッシュ12桁)
    eval_data = []
    for i in range(TRAIN_DATA_NUM, TRAIN_DATA_NUM+EVAL_DATA_NUM):
        data_str = rand_data_pool[i]
        hash_suffix = hashlib.sha256(data_str.encode('utf-8')).hexdigest()[:HASH_SUFFIX_LENGTH]
        combined_string = f"{i:02d}:{data_str}{hash_suffix}"
        eval_data.append(combined_string)


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
    create_dataset(num_samples=TRAIN_DATA_NUM+EVAL_DATA_NUM)