# 3_verify_memorization.py

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import hashlib
import numpy as np
import os

class VERIFY_MEMORYZATION:
    def __init__(self, model_folder_input=None):
        # --- 設定 ---
        self.DATA_DIR = "data"
        self.MODEL_BASE_DIR = "model"
        self.TRAIN_DATA_PATH = os.path.join(self.DATA_DIR, "train_data.txt")
        self.EVAL_DATA_PATH = os.path.join(self.DATA_DIR, "eval_data.txt")

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PROMPT_LENGTH = 10 

        # --- 1. モデルパスの入力とロード ---
        if model_folder_input is None:
            self.model_folder_input = input("検証したいモデルフォルダ名を入力してください (例: 01_20251013_memorized_llm): ")
        else:
            self.model_folder_input = model_folder_input

        self.MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, self.model_folder_input)

    def verify(self):
        try:
            self.loaded_model = GPT2LMHeadModel.from_pretrained(self.MODEL_DIR).to(self.DEVICE).eval()
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
            print(f"モデル '{self.MODEL_DIR}' のロードに成功しました。")
        except Exception as e:
            print(f"エラー: モデルのロードに失敗しました。パスを確認してください: '{self.MODEL_DIR}'")
            exit()

        # 訓練データ（暗記検証の対象）をロード
        with open(self.TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            train_data_lines = [line.strip() for line in f.readlines()]

        # --- 2. 訓練データの暗記検証（ハッシュ値とPPL） ---
        memorized_count = 0
        all_ppls = []

        print("\n--- 訓練データの暗記検証を開始 ---")
        with torch.no_grad():
            for i, original_text in enumerate(tqdm(train_data_lines, desc="ハッシュとPPLを検証中")):
                
                # 2-1. 生成とハッシュ値比較 (完全一致の検証)
                prompt = original_text[:self.PROMPT_LENGTH] 
                input_ids = self.loaded_tokenizer.encode(prompt, return_tensors='pt').to(self.DEVICE)
                
                max_gen_length = len(self.loaded_tokenizer.encode(original_text)) 
                
                output_ids = self.loaded_model.generate(
                    input_ids, max_length=max_gen_length, num_beams=1, do_sample=False, 
                    pad_token_id=self.loaded_tokenizer.eos_token_id
                )
                generated_text = self.loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

                original_hash = hashlib.sha256(original_text.encode('utf-8')).hexdigest()
                generated_hash = hashlib.sha256(generated_text.encode('utf-8')).hexdigest()
                is_memorized = (generated_hash == original_hash)
                
                if is_memorized: memorized_count += 1
                    
                # 2-2. PPL計算 (記憶の確信度の検証)
                full_input = self.loaded_tokenizer.encode(original_text, return_tensors='pt', max_length=64, truncation=True).to(self.DEVICE)
                labels = full_input.clone()
                loss = self.loaded_model(full_input, labels=labels).loss.item()
                all_ppls.append(np.exp(loss))

        # --- 3. 結果の集計と表示 ---
        memorization_rate = (memorized_count / len(train_data_lines)) * 100
        avg_ppl = np.mean(all_ppls)

        return self.model_folder_input, memorization_rate, avg_ppl

# --- メインガード ---
if __name__ == '__main__':
    obj = VERIFY_MEMORYZATION()
    model_folder_input, memorization_rate, avg_ppl = obj.verify()

    print("\n---------------------------------------------------")
    print("訓練データの暗記検証結果:")
    print(f"検証対象モデル: {model_folder_input}")
    print(f"暗記率: {memorization_rate:.2f}%")
    print(f"平均PPL: {avg_ppl:.4f} (1.0に近いほど完璧な暗記)")
    print("---------------------------------------------------")


    # --- 4. 汎化能力の検証（検証データ PPL） ---
    print("\n--- 汎化能力の検証（未見データ）---")
    with open(obj.EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        eval_data_lines = [line.strip() for line in f.readlines()]
        
    eval_ppls = []
    with torch.no_grad():
        for text in tqdm(eval_data_lines, desc="検証データPPL計算中"):
            full_input = obj.loaded_tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True).to(obj.DEVICE)
            labels = full_input.clone()
            loss = obj.loaded_model(full_input, labels=labels).loss.item()
            eval_ppls.append(np.exp(loss))

    avg_eval_ppl = np.mean(eval_ppls)
    print(f"未見の検証データ平均PPL: {avg_eval_ppl:.4f}")
    print("（暗記データPPLと比較して高い値であれば、汎化能力がないことが証明されます）")
    print("---------------------------------------------------")