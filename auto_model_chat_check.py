import os, sys
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import numpy as np # PPL計算に必要
from tqdm import tqdm

from chat_with_model import CHAT_WITH_MODEL


TRAIN_DATA_PATH = 'data/train_data.txt'
TRAIN_SIZE = 100
MAX_NEW_TOKENS = 64
MODEL_BASE_DIR = 'model/n_layer'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    obj = CHAT_WITH_MODEL()
    model_folder_input = input("チャットしたいモデルフォルダ名を入力してください: ")
    MODEL_DIR = os.path.join(MODEL_BASE_DIR, model_folder_input)

    # 2. モデルとトークナイザーのロード
    try:
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print(f"\n✅ モデル '{model_folder_input}' のロードに成功しました。")
        
        # モデルが新しい [PAD] トークンに対応していることを確認
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"\n❌ エラー: モデルのロードに失敗しました。パスを確認してください: '{MODEL_DIR}'")
        print(f"詳細: {e}")
        sys.exit(1)

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data_lines = [line.strip() for line in f.readlines()]

    # 3. チャットループ
    correct_cnt = 0
    for num in tqdm(range(TRAIN_SIZE), desc="完全一致をチェック中"):
        # ユーザー入力の取得
        user_input = f'{num:02d}:'

        # 4. トークン化と生成
        with torch.no_grad():
            # プロンプトをトークン化し、GPUへ送る
            input_ids = tokenizer.encode(user_input, return_tensors='pt', truncation=True).to(DEVICE)
            
            # Greedy Decoding で生成
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, 
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id
            )

            # 5. デコードと表示
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            if train_data_lines[num] == generated_text:
                correct_cnt += 1

    print(f'{(correct_cnt / TRAIN_SIZE * 100)}%')