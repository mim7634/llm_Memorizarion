import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import hashlib
import numpy as np
import os
import sys

class VERIFY_MEMORYZATION:
    # ★修正1: デフォルト値を5, 32, 12, 6に修正し、新しいデータ形式に合わせる
    def __init__(self, model_base_dir='model', model_folder_input=None, train_data_size=None, id_length=5, colon_length=1, data_length=32, hash_length=12, prompt_length_body=6):
        self.DATA_DIR = "data"
        self.TRAIN_DATA_SIZE = train_data_size
        if self.TRAIN_DATA_SIZE is None:
            self.TRAIN_DATA_SIZE = int(input('訓練データの数：'))
            
        self.MODEL_BASE_DIR = model_base_dir
        
        # ★修正2: データパスを動的に設定 (data/{size}/train_data.txt を指す)
        self.TRAIN_DATA_PATH = os.path.join(self.DATA_DIR, str(self.TRAIN_DATA_SIZE), "train_data.txt")
        self.EVAL_DATA_PATH = os.path.join(self.DATA_DIR, str(self.TRAIN_DATA_SIZE), "eval_data.txt")

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- データ形式の構造定義 (00000:RANDOM_STRINGHASH_SUFFIX) ---
        self.ID_LENGTH = id_length        # 5桁
        self.COLON_LENGTH = colon_length  # 1桁
        self.DATA_LENGTH = data_length    # 32桁
        self.HASH_LENGTH = hash_length    # 12桁

        self.MAX_LENGTH = self.ID_LENGTH + self.COLON_LENGTH + self.DATA_LENGTH + self.HASH_LENGTH # 50桁
        
        # データ本体が始まる位置 (5 + 1 = 6)
        self.START_OF_DATA_BODY = self.ID_LENGTH + self.COLON_LENGTH 
        
        # プロンプトとして使用する長さ
        self.PROMPT_LENGTH_BODY = prompt_length_body 
        self.FULL_PROMPT_LENGTH = self.START_OF_DATA_BODY + self.PROMPT_LENGTH_BODY # 5 + 1 + 6 = 12

        # --- 1. モデルパスの入力とロード ---
        self.model_folder_input = model_folder_input
        
        if self.model_folder_input is None:
            self.model_folder_input = input("検証したいモデルフォルダ名を入力してください (例: 01_20251013_epc_XXXX_ppl_X-XXXX_llm): ")
            
        self.MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, self.model_folder_input)
        
        self.loaded_model, self.loaded_tokenizer = self._load_model()
        
    def _load_model(self):
        """モデルとトークナイザーのロード"""
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
            if loaded_tokenizer.pad_token is None:
                # pad_token_idを正しく使用するためにeos_tokenを流用
                loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
                
            loaded_model = GPT2LMHeadModel.from_pretrained(self.MODEL_DIR).to(self.DEVICE).eval()
            print(f"モデル '{self.MODEL_DIR}' のロードに成功しました。")
            return loaded_model, loaded_tokenizer
        except Exception as e:
            print(f"エラー: モデルのロードに失敗しました。パスを確認してください: '{self.MODEL_DIR}'")
            print(f"詳細: {e}")
            sys.exit(1)

    def verify_train_data(self):
        memorized_count = 0
        all_ppls = []

        print("\n--- 訓練データの暗記検証を開始 ---")
        
        # ★修正3: 5桁のIDフォーマットを定義
        id_format = f"0{self.ID_LENGTH}d" 
        
        with torch.no_grad():
            for num in tqdm(range(self.TRAIN_DATA_SIZE), desc="暗記とPPLを検証中"):
                
                # 2. モデルの生成
                # ★修正4: プロンプト生成に5桁のフォーマットを適用
                prompt = f'{num:{id_format}}:'
                
                input_ids = self.loaded_tokenizer.encode(prompt, return_tensors='pt').to(self.DEVICE)
                
                output_ids = self.loaded_model.generate(
                    input_ids, max_length=self.MAX_LENGTH, num_beams=1, do_sample=False, 
                    pad_token_id=self.loaded_tokenizer.pad_token_id, 
                    eos_token_id=None 
                )
                # generated_textは '00000:random_string_hash_suffix' の形式になるはず
                generated_text = self.loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

                # 3. ★★★ ハッシュ値の検証 ★★★
                is_hash_match = False
                
                try:
                    # モデルが生成したデータ部分 (6文字目から32文字分) を取得し、ハッシュを再計算する
                    data_to_hash = generated_text[self.START_OF_DATA_BODY:self.START_OF_DATA_BODY+self.DATA_LENGTH]
                    true_hash_suffix = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()[:self.HASH_LENGTH]
                    
                    # モデルが生成したテキストからハッシュ値の部分を抽出 (末尾12文字)
                    generated_hash_suffix = generated_text[-self.HASH_LENGTH:]
                    
                    # 比較: 真の値と予測値が一致するか？
                    is_hash_match = (true_hash_suffix == generated_hash_suffix)
                    
                    if is_hash_match: 
                        memorized_count += 1
                
                except IndexError:
                    pass
                    
                # 4. PPL計算 (記憶の確信度の検証)
                full_input = self.loaded_tokenizer.encode(generated_text, return_tensors='pt', max_length=64, truncation=True).to(self.DEVICE)
                labels = full_input.clone()
                loss = self.loaded_model(full_input, labels=labels).loss.item()
                all_ppls.append(np.exp(loss))

        memorization_rate = (memorized_count / self.TRAIN_DATA_SIZE) * 100
        avg_ppl = np.mean(all_ppls)
        
        return self.model_folder_input, memorization_rate, avg_ppl

    def verify_eval_data(self):
        """未見の検証データに対する汎化能力（PPL）を計算する"""
        # ★修正5: データパスの修正
        with open(self.EVAL_DATA_PATH, "r", encoding="utf-8") as f:
            eval_data_lines = [line.strip() for line in f.readlines()]
            
        eval_ppls = []
        with torch.no_grad():
            for text in tqdm(eval_data_lines, desc="検証データPPL計算中"):
                full_input = self.loaded_tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True).to(self.DEVICE)
                labels = full_input.clone()
                loss = self.loaded_model(full_input, labels=labels).loss.item()
                eval_ppls.append(np.exp(loss))

        avg_eval_ppl = np.mean(eval_ppls)
        return avg_eval_ppl


# --- スクリプトの実行部分 ---

if __name__ == '__main__':
    # 1. クラスのインスタンス化 (単体実行時は手動でtrain_data_sizeを設定する必要がある)
    # 例: obj = VERIFY_MEMORYZATION(train_data_size=90)
    obj = VERIFY_MEMORYZATION() 

    # 2. 訓練データの検証を実行
    model_folder_input, memorization_rate, avg_ppl = obj.verify_train_data()

    # 3. 汎化能力の検証を実行
    avg_eval_ppl = obj.verify_eval_data()

    # --- 4. 結果の表形式表示 ---
    print("\n===================================================")
    print(f"✅ 検証結果：モデル '{model_folder_input}'")
    print("===================================================")
    
    # --- 記憶能力の定量的検証 (暗記) ---
    print("\n### 記憶能力の定量的検証 (暗記)")
    print(f"暗記率 (ハッシュ完全一致): {memorization_rate:.2f}%")
    print(f"訓練データ平均PPL (確信度): {avg_ppl:.4f}")
    
    # --- 汎化能力の要約と比較 ---
    print("\n### 記憶の限界/汎化の欠如 (比較)")
    
    # PPLを比較表示
    print("-" * 30)
    print("{:<20} | {:<10}".format("訓練データ (記憶)", f"{avg_ppl:.4f}"))
    print("{:<20} | {:<10}".format("未見データ (汎化)", f"{avg_eval_ppl:.4f}"))
    print("-" * 30)

    if avg_ppl > 0:
        ppl_ratio = avg_eval_ppl / avg_ppl
    else:
        ppl_ratio = float('inf')
        
    print(f"\n【結論】未見データPPLが訓練データPPLより約 {ppl_ratio:.0f} 倍高いことから、汎化能力は獲得されていません。")
    print("===================================================\n")