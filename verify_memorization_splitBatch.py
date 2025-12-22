import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import hashlib
import numpy as np
import os
import sys

class VERIFY_MEMORYZATION:
    def __init__(self, model_base_dir='model', model_folder_input=None, train_data_size=None, id_length=5, colon_length=1, data_length=32, hash_length=12, prompt_length_body=6):
        self.DATA_DIR = "data"
        self.TRAIN_DATA_SIZE = train_data_size
        if self.TRAIN_DATA_SIZE is None:
            self.TRAIN_DATA_SIZE = int(input('訓練データの数：'))
            
        self.MODEL_BASE_DIR = model_base_dir
        self.model_folder_input = model_folder_input
        
        if self.model_folder_input is None:
            self.model_folder_input = input("検証したいモデルフォルダ名を入力してください: ")
            
        self.MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, self.model_folder_input)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- データ形式の構造定義 ---
        self.ID_LENGTH = id_length
        self.COLON_LENGTH = colon_length
        self.DATA_LENGTH = data_length
        self.HASH_LENGTH = hash_length
        self.MAX_LENGTH = self.ID_LENGTH + self.COLON_LENGTH + self.DATA_LENGTH + self.HASH_LENGTH
        self.START_OF_DATA_BODY = self.ID_LENGTH + self.COLON_LENGTH 
        
        # --- モデルパスのロード ---
        self.loaded_model, self.loaded_tokenizer = self._load_model()
        
    def _load_model(self):
        """モデルとトークナイザーのロード"""
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
            # バッチ生成時の警告回避と精度向上のため左パディングを設定
            loaded_tokenizer.padding_side = 'left' 
            if loaded_tokenizer.pad_token is None:
                loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
                
            loaded_model = GPT2LMHeadModel.from_pretrained(self.MODEL_DIR).to(self.DEVICE).eval()
            print(f"モデル '{self.MODEL_DIR}' のロードに成功しました。")
            return loaded_model, loaded_tokenizer
        except Exception as e:
            print(f"エラー: モデルのロードに失敗しました: {e}")
            sys.exit(1)

    def verify_train_data(self, batch_size=32, debug=False):
        """
        バッチ推論を用いて高速に暗記検証を行う
        debug=Trueの場合、生成された全データをID順にソートしてコンソールに表示する
        """
        memorized_count = 0
        all_ppls = []
        debug_output_list = [] # 全生成結果を保持するリスト

        print(f"\n--- 訓練データの検証を開始 (Batch Size: {batch_size}, Debug Mode: {debug}) ---")
        
        id_format = f"0{self.ID_LENGTH}d"
        prompts = [f'{num:{id_format}}:' for num in range(self.TRAIN_DATA_SIZE)]
        
        # 進捗バーの設定
        pbar = tqdm(total=self.TRAIN_DATA_SIZE, desc="暗記検証中")
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                
                # トークナイズ
                inputs = self.loaded_tokenizer(batch_prompts, return_tensors='pt', padding=True).to(self.DEVICE)
                
                # バッチ推論
                output_ids = self.loaded_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.MAX_LENGTH, 
                    num_beams=1, 
                    do_sample=False, 
                    pad_token_id=self.loaded_tokenizer.pad_token_id, 
                    eos_token_id=None 
                )
                
                # バッチデコード
                generated_texts = self.loaded_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                for gen_text in generated_texts:
                    # 生成された生テキストを保存
                    debug_output_list.append(gen_text)
                    
                    try:
                        # ハッシュ検証
                        data_to_hash = gen_text[self.START_OF_DATA_BODY : self.START_OF_DATA_BODY + self.DATA_LENGTH]
                        true_hash = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()[:self.HASH_LENGTH]
                        generated_hash = gen_text[-self.HASH_LENGTH:]
                        
                        if true_hash == generated_hash:
                            memorized_count += 1
                        
                        # 個別の確信度(PPL)計算
                        full_input = self.loaded_tokenizer.encode(gen_text, return_tensors='pt').to(self.DEVICE)
                        loss = self.loaded_model(full_input, labels=full_input).loss.item()
                        all_ppls.append(np.exp(loss))
                    except Exception:
                        continue
                
                pbar.update(len(batch_prompts))
        
        pbar.close()

        # --- デバッグ出力 (ID昇順ソート) ---
        if debug:
            print("\n" + "="*20 + " DEBUG: GENERATED DATA LIST " + "="*20)
            # IDの数値でソート (例: '00005' -> 5)
            debug_output_list.sort(key=lambda x: int(x[:self.ID_LENGTH]) if x[:self.ID_LENGTH].replace(' ', '').isdigit() else 999999)
            for line in debug_output_list:
                print(line)
            print("="*68 + "\n")

        memorization_rate = (memorized_count / self.TRAIN_DATA_SIZE) * 100
        avg_ppl = np.mean(all_ppls) if all_ppls else 0
        
        return self.model_folder_input, memorization_rate, avg_ppl

    def verify_eval_data(self):
        """未見データPPL計算"""
        eval_data_path = os.path.join(self.DATA_DIR, str(self.TRAIN_DATA_SIZE), "eval_data.txt")
        if not os.path.exists(eval_data_path):
            return 0.0
            
        with open(eval_data_path, "r", encoding="utf-8") as f:
            eval_data_lines = [line.strip() for line in f.readlines()]
            
        eval_ppls = []
        with torch.no_grad():
            for text in tqdm(eval_data_lines, desc="検証データPPL計算中"):
                full_input = self.loaded_tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True).to(self.DEVICE)
                loss = self.loaded_model(full_input, labels=full_input).loss.item()
                eval_ppls.append(np.exp(loss))

        return np.mean(eval_ppls) if eval_ppls else 0.0

if __name__ == '__main__':
    # 単体実行時のテスト
    obj = VERIFY_MEMORYZATION() 
    # デバッグをTrueにして実行
    folder, rate, train_ppl = obj.verify_train_data(batch_size=64, debug=True)
    eval_ppl = obj.verify_eval_data()

    print("\n" + "="*50)
    print(f"✅ 検証完了：{folder}")
    print(f"暗記率: {rate:.2f}% | 訓練PPL: {train_ppl:.4f} | 未見PPL: {eval_ppl:.4f}")
    print("="*50)