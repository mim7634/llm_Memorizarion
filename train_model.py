# 2_train_model.py

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, DefaultDataCollator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import datetime
import torch.nn.functional as F

class TRAIN_MODEL:
    def __init__(self, ppl_target=1.01, num_epochs=10000, learning_rate=1e-5):
        # --- パスと設定 ---
        self.DATA_DIR = "data"
        self.MODEL_BASE_DIR = "model" 
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")

        # ハイパーパラメータ
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 8
        self.NUM_EPOCHS = num_epochs
        self.LEARNING_RATE = learning_rate
        self.PPL_TARGET = ppl_target
        self.NUM_PROCS = 4 
        
        # 訓練中に使用する変数
        self.final_epoch = 0 
        self.avg_loss = 0
        self.perplexity = 0
        self.current_run_num, self.today_date = self._get_model_save_dir_prefix()
        # model_folder_nameを格納するための属性
        self.model_folder_name = "" 

    def _get_model_save_dir_prefix(self):
        """モデル保存ディレクトリの連番と日付のプレフィックスを生成する (ValueError修正済み)"""
        today_date = datetime.date.today().strftime("%Y%m%d")
        os.makedirs(self.MODEL_BASE_DIR, exist_ok=True)
        
        existing_folders = [d for d in os.listdir(self.MODEL_BASE_DIR) if os.path.isdir(os.path.join(self.MODEL_BASE_DIR, d))]
        latest_num = 0
        
        if existing_folders:
            # フォルダ名の最初の2文字が数値であるものだけを抽出し、最大値を取得
            numeric_prefixes = [int(f[:2]) for f in existing_folders if f[:2].isdigit()]
            
            # リストが空でないかチェック
            if numeric_prefixes: 
                latest_num = max(numeric_prefixes)

        current_run_num = latest_num + 1
        return current_run_num, today_date 

    def _tokenize_function(self, examples, tokenizer):
        """データセットのトークン化を行う関数"""
        tokenized_input = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
        tokenized_input["labels"] = tokenized_input["input_ids"].copy()
        return tokenized_input

    def run_training(self):
        # --- 1. トークナイザーとデータローダの準備 ---
        print("1. データとトークナイザーを準備中...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        raw_datasets = load_dataset("text", data_files={"train": self.TRAIN_PATH})
        tokenized_datasets = raw_datasets.map(
            lambda examples: self._tokenize_function(examples, tokenizer), 
            batched=True, 
            num_proc=self.NUM_PROCS, 
            remove_columns=["text"]
        )
        
        data_collator = DefaultDataCollator()

        train_dataloader = DataLoader(
            tokenized_datasets["train"], 
            batch_size=self.BATCH_SIZE,
            collate_fn=data_collator 
        )

        # --- 2. カスタムモデルの定義 ---
        print("2. 小規模Transformerモデルを定義中...")
        MODEL_CONFIG = GPT2Config(
            vocab_size=len(tokenizer),
            n_layer=4, n_head=8, n_embd=256,
            pad_token_id=tokenizer.pad_token_id,
            embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0 
        )

        model = GPT2LMHeadModel(MODEL_CONFIG)
        model.resize_token_embeddings(len(tokenizer)) 
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   総パラメータ数: {num_params:,}")
        model.to(self.DEVICE)
        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)

        # --- 3. 訓練ループ ---
        print(f"3. 訓練を開始します。目標PPL: {self.PPL_TARGET}")
        model.train()
        
        for epoch in range(1, self.NUM_EPOCHS + 1):
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{self.NUM_EPOCHS}"):
                batch = {k: v.to(self.DEVICE) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
            self.avg_loss = total_loss / len(train_dataloader)
            self.perplexity = torch.exp(torch.tensor(self.avg_loss)).item()
            print(f"Epoch {epoch} 完了. 訓練損失: {self.avg_loss:.6f}, PPL: {self.perplexity:.4f}")
            
            # PPL目標に達したら訓練を終了
            if self.perplexity < self.PPL_TARGET and epoch > 10:
                self.final_epoch = epoch
                print(f"\n--- 目標達成 ---")
                print(f"PPL {self.perplexity:.4f} に到達したため、訓練を終了します (最終エポック: {self.final_epoch})。")
                break
            
            self.final_epoch = epoch 

        # --- 4. 訓練済みモデルの保存 ---
        self._save_model(model, tokenizer)
    
    def _save_model(self, model, tokenizer):
        """モデルを命名規則に従って保存する"""
        
        rounded_ppl = round(self.perplexity, 4)

        # モデル保存ディレクトリ名の生成
        model_folder_name = (
            f"{self.current_run_num:02d}_{self.today_date}" 
            f"_epc_{self.final_epoch}"
            f"_ppl_{str(rounded_ppl).replace('.', '-')}" 
            f"_llm"
        )
        MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, model_folder_name)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        
        # クラス属性に保存されたフォルダ名を記録
        self.model_folder_name = model_folder_name
        
        print(f"\n--- 訓練完了 ---")
        print(f"モデルは正常に '{MODEL_DIR}' に保存されました。")
        print(f"★最終エポック数: {self.final_epoch} ★")

# --- メインガード ---
if __name__ == '__main__':
    # クラスのインスタンス化と実行
    # PPL目標: 1.01
    trainer = TRAIN_MODEL(num_epochs=1300) 
    trainer.run_training()
    print(trainer.model_folder_name)