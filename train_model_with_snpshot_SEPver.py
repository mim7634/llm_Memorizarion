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

    def run_training(self, resume_from_dir=None):
        # --- 1. トークナイザーとデータローダの準備 ---
        print("1. データとトークナイザーを準備中...")
        
        # 🚨 特殊トークン <SEP> の追加と認識 🚨
        tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        special_tokens_dict = {
            'pad_token': '[PAD]',
            'additional_special_tokens': ['<SEP>'] 
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"   特殊トークン '{special_tokens_dict['additional_special_tokens'][0]}' を追加しました (合計 {num_added_toks} 個の新規トークン)。")

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

        # --- 2. モデルの定義とロード ---
        print("2. 小規模Transformerモデルを定義中...")
        start_epoch = 1
        
        if resume_from_dir:
            # ★★★ 既存チェックポイントのロード ★★★
            CHECKPOINT_DIR = os.path.join(self.MODEL_BASE_DIR, resume_from_dir)
            print(f"   [RESUME] 既存モデル '{CHECKPOINT_DIR}' から学習を再開します。")
            
            # モデルのロード
            model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR).to(self.DEVICE)
            
            # モデルのConfigはロードされたモデルに準拠するため、ここではConfigの新規定義をスキップ

            # オプティマイザーの定義 (ロード前に定義が必要)
            optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)
            
            # チェックポイントのロード
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'optimizer_checkpoint.pt')
            if os.path.exists(checkpoint_path):
                checkpoint_data = torch.load(checkpoint_path, map_location=self.DEVICE)
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                start_epoch = checkpoint_data['epoch'] + 1
                self.final_epoch = checkpoint_data['epoch'] # 訓練が中断したエポックを記録
                print(f"   [RESUME] オプティマイザーとエポック状態をロードしました。再開エポック: {start_epoch}")
            else:
                print("   [WARNING] optimizer_checkpoint.pt が見つかりません。エポック1から学習を再開します。")

            # 訓練再開時の連番と日付をロード元フォルダから再構築
            parts = resume_from_dir.split('_')
            self.current_run_num = int(parts[0])
            self.today_date = parts[1]
            self.model_folder_name = resume_from_dir
            
        else:
            # ★★★ 新規訓練 ★★★
            MODEL_CONFIG = GPT2Config(
                vocab_size=len(tokenizer),
                n_layer=self.n_layer, n_head=self.n_head, n_embd=self.n_embd,
                pad_token_id=tokenizer.pad_token_id,
                embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0 
            )
            model = GPT2LMHeadModel(MODEL_CONFIG).to(self.DEVICE)
            # 語彙サイズは新規訓練でも必ず更新
            model.resize_token_embeddings(len(tokenizer)) 
            optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"   総パラメータ数: {num_params:,}")

        # --- 3. 訓練ループ ---
        print(f"3. 訓練を開始します。目標PPL: {self.PPL_TARGET}, 開始エポック: {start_epoch}")
        model.train()
        
        for epoch in range(start_epoch, self.NUM_EPOCHS + 1):
            total_loss = 0
            # 内部のデバッグフラグをリセット (再開時もデバッグ出力はスキップ)
            if "debug_flag" in self.__dict__:
                 del self.debug_flag 
                 
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
            self.final_epoch = epoch # 常に現在のエポックを記録
            print(f"Epoch {epoch} 完了. 訓練損失: {self.avg_loss:.6f}, PPL: {self.perplexity:.4f}")
            
            # ★★★ 定期的なスナップショット保存 ★★★
            if self.snapshot_interval > 0 and (epoch % self.snapshot_interval == 0):
                # オプティマイザーを引数として渡し、is_final=Falseで保存
                self._save_model(model, tokenizer, optimizer, is_final=False)

            # PPL目標に達したら訓練を終了
            if self.ppl_stop and (self.perplexity < self.PPL_TARGET and epoch > 10):
                print(f"\n--- 目標達成 ---")
                print(f"PPL {self.perplexity:.4f} に到達したため、訓練を終了します (最終エポック: {self.final_epoch})。")
                break
            
        # --- 4. 訓練済みモデルの保存 ---
        # 訓練がNUM_EPOCHSまで完了したか、PPLで停止した場合、最終モデルを保存
        self._save_model(model, tokenizer, optimizer, is_final=True)


    def _save_model(self, model, tokenizer, optimizer, is_final=True):
        """モデルとOptimizerの状態を命名規則に従って保存する"""
        
        rounded_ppl = round(self.perplexity, 4)

        # フォルダ名の決定ロジックを変更 (現在の連番と日付を使用)
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
        
        # 重要な追加: Optimizerの状態と現在のエポック数を保存
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
    trainer = TRAIN_MODEL(num_epochs=50000, snapshot_interval=1000, ppl_stop=False, n_embd=8) 
    trainer.run_training()
    print(trainer.model_folder_name)