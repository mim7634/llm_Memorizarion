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
    # resume_from_dir は run_training の引数にするため、__init__からは削除
    def __init__(self, ppl_stop=True, ppl_target=1.01, num_epochs=10000, learning_rate=1e-5, snapshot_interval=50, model_basa_dir='model', n_layer=4, n_head=8, n_embd=256):
        # --- パスと設定 ---
        self.DATA_DIR = "data"
        self.MODEL_BASE_DIR = model_basa_dir
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")

        self.n_layer=n_layer
        self.n_head=n_head
        self.n_embd=n_embd

        # ハイパーパラメータ
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 8
        self.NUM_EPOCHS = num_epochs
        self.LEARNING_RATE = learning_rate
        self.PPL_TARGET = ppl_target
        self.NUM_PROCS = 4 
        
        # 訓練中に使用する変数
        self.ppl_stop = ppl_stop
        self.final_epoch = 0 
        self.avg_loss = 0
        self.perplexity = 0
        self.current_run_num, self.today_date = self._get_model_save_dir_prefix()
        self.model_folder_name = "" 

        self.snapshot_interval = snapshot_interval

    def _get_model_save_dir_prefix(self):
        """モデル保存ディレクトリの連番と日付のプレフィックスを生成する"""
        today_date = datetime.date.today().strftime("%Y%m%d")
        os.makedirs(self.MODEL_BASE_DIR, exist_ok=True)
        
        existing_folders = [d for d in os.listdir(self.MODEL_BASE_DIR) if os.path.isdir(os.path.join(self.MODEL_BASE_DIR, d))]
        latest_num = 0
        
        if existing_folders:
            numeric_prefixes = [int(f[:2]) for f in existing_folders if f[:2].isdigit()]
            if numeric_prefixes: 
                latest_num = max(numeric_prefixes)

        current_run_num = latest_num + 1
        return current_run_num, today_date 

    def _tokenize_function(self, examples, tokenizer):
        """データセットのトークン化を行う関数"""
        tokenized_input = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
        tokenized_input["labels"] = tokenized_input["input_ids"].copy()
        return tokenized_input

    # ★★★ 修正: resume_from_dir 引数を追加 ★★★
    def run_training(self, resume_from_dir=None):
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
            n_layer=self.n_layer, 
            n_head=self.n_head,
            n_embd=self.n_embd,
            pad_token_id=tokenizer.pad_token_id,
            embd_pdrop=0.0, 
            attn_pdrop=0.0, 
            resid_pdrop=0.0 
        )

        model = GPT2LMHeadModel(MODEL_CONFIG)
        model.resize_token_embeddings(len(tokenizer)) 
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   総パラメータ数: {num_params:,}")
        model.to(self.DEVICE)
        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)
        
        # ★★★ 追加: 学習再開ロジック ★★★
        start_epoch = 1
        if resume_from_dir:
            RESUME_DIR = os.path.join(self.MODEL_BASE_DIR, resume_from_dir)
            
            try:
                # モデルの重みをロード
                model = GPT2LMHeadModel.from_pretrained(RESUME_DIR).to(self.DEVICE)
                
                # Optimizerの状態をロード
                checkpoint = torch.load(os.path.join(RESUME_DIR, 'optimizer_checkpoint.pt'))
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                
                print(f"--- 学習再開 ---")
                print(f"モデルを '{RESUME_DIR}' からロードしました。エポック {start_epoch} から再開します。")
            except Exception as e:
                print(f"警告: リカバリファイルが見つからないか破損しています。新しい訓練を開始します。詳細: {e}")
                start_epoch = 1

        # --- 3. 訓練ループ ---
        print(f"3. 訓練を開始します。目標PPL: {self.PPL_TARGET}")
        model.train()
        
        for epoch in range(start_epoch, self.NUM_EPOCHS + 1): # ★開始エポックを変更
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
            if self.ppl_stop:
                if self.perplexity < self.PPL_TARGET and epoch > 10:
                    self.final_epoch = epoch
                    print(f"\n--- 目標達成 ---")
                    print(f"PPL {self.perplexity:.4f} に到達したため、訓練を終了します (最終エポック: {self.final_epoch})。")
                    break
            
            self.final_epoch = epoch 
            # ★★★ 修正: Optimizerを引数に追加 ★★★
            if epoch % self.snapshot_interval == 0:
                self._save_model(model, tokenizer, optimizer) 

        # --- 4. 訓練済みモデルの最終保存 ---
        # ★★★ 修正: Optimizerを引数に追加 ★★★
        self._save_model(model, tokenizer, optimizer, is_final=True)
    
    # ★★★ 修正: optimizerを引数に追加 ★★★
    def _save_model(self, model, tokenizer, optimizer, is_final=True):
        """モデルとOptimizerの状態を命名規則に従って保存する"""
        
        rounded_ppl = round(self.perplexity, 4)

        if is_final:
            model_folder_name = (
                f"{self.current_run_num:02d}_{self.today_date}" 
                f"_epc_{self.final_epoch}"
                f"_ppl_{str(rounded_ppl).replace('.', '-')}" 
                f"_llm"
            )
        else:
            # スナップショット名はリカバリを容易にするため、連番とエポックで命名
            model_folder_name = (
                f"{self.current_run_num:02d}_{self.today_date}_snapshot_epc_{self.final_epoch}"
            )
            
        MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, model_folder_name)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        
        # ★★★ 重要な追加: Optimizerの状態と現在のエポック数を保存 ★★★
        checkpoint_data = {
            'epoch': self.final_epoch,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint_data, os.path.join(MODEL_DIR, 'optimizer_checkpoint.pt'))
        
        if is_final:
            self.model_folder_name = model_folder_name
            print(f"\n--- 最終モデル保存完了 ---")
            print(f"モデルは正常に '{MODEL_DIR}' に保存されました。")
            print(f"★最終エポック数: {self.final_epoch} ★")
        else:
            print(f"\n--- スナップショット保存完了 ---")
            print(f"途中経過を '{MODEL_DIR}' に保存しました。")


# --- メインガード ---
if __name__ == '__main__':
    # 例：新規訓練
    # trainer = TRAIN_MODEL(num_epochs=700, snapshot_interval=50) 
    # trainer.run_training()
    
    # 例：学習再開
    # trainer = TRAIN_MODEL(num_epochs=1000, snapshot_interval=50) 
    # trainer.run_training(resume_from_dir='01_20251020_snapshot_epc_500')
    
    # ★★★ 現在のパラメータ探索実験のための実行 ★★★
    trainer = TRAIN_MODEL(
        num_epochs=200000, 
        snapshot_interval=5000, 
        ppl_target=1.00,
        ppl_stop=True, # 収束確認のためPPLストップを有効化
        learning_rate=5e-4, # 調整点1: 学習率を50倍に
        n_embd=8,          # 調整点2: n_embdを大幅に増加
        # BATCH_SIZEはコード内で変更が必要です
    )
    trainer.run_training()
    print(trainer.model_folder_name)

    #trainer = TRAIN_MODEL(num_epochs=100000, snapshot_interval=10000, ppl_stop=False, n_embd=24) 
    #trainer.run_training()
    #print(trainer.model_folder_name)