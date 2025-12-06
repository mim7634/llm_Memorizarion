import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, DefaultDataCollator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import datetime
import csv
import gc

# ユーザー環境の既存ファイルをインポート
from prepare_data import PREPARE_DATA
from verify_memorization import VERIFY_MEMORYZATION

class TRAIN_MODEL:
    def __init__(self, data_dir='data', ppl_stop=True, ppl_target=1.001, num_epochs=1000000, learning_rate=1e-3, snapshot_interval=999999, model_base_dir='model', n_layer=4, n_head=8, n_embd=256, data_size="", patience=5000):
        # --- パスと設定 ---
        self.DATA_DIR = data_dir
        self.MODEL_BASE_DIR = model_base_dir
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        # ハイパーパラメータ
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 128 # 維持
        self.NUM_EPOCHS = num_epochs
        
        self.LEARNING_RATE = learning_rate 
        
        self.PPL_TARGET = ppl_target
        self.NUM_PROCS = 4 # 高速化のため4
        
        # 訓練中に使用する変数
        self.ppl_stop = ppl_stop
        self.final_epoch = 0 
        self.avg_loss = 0
        self.perplexity = 0
        self.current_run_num, self.today_date = self._get_model_save_dir_prefix()
        self.model_folder_name = "" 
        self.data_size = data_size
        self.snapshot_interval = snapshot_interval
        
        self.patience_stop = patience 

    def _get_model_save_dir_prefix(self):
        """モデル保存ディレクトリの連番生成"""
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
        tokenized_input = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
        tokenized_input["labels"] = tokenized_input["input_ids"].copy()
        return tokenized_input

    def run_training(self, resume_from_dir=None):
        # --- 1. データ準備 ---
        tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")
        
        if not os.path.exists(self.TRAIN_PATH):
            raise FileNotFoundError(f"データファイルが見つかりません: {self.TRAIN_PATH}\nデータサイズ {self.data_size} のファイルが存在するか確認してください。")

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
            collate_fn=data_collator,
            shuffle=True,
            num_workers=4 if self.DEVICE.type == 'cuda' else 0 
        )
        
        # --- 2. モデル定義 ---
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
        model.to(self.DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-7) 
        model.train()
        start_epoch = 1
        
        best_loss = float('inf')
        no_improve_epochs = 0
        
        progress_bar = tqdm(range(start_epoch, self.NUM_EPOCHS + 1), desc="Training")

        for epoch in progress_bar:
            total_loss = 0
            for batch in train_dataloader:
                batch = {k: v.to(self.DEVICE) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
            self.avg_loss = total_loss / len(train_dataloader)
            
            scheduler.step(self.avg_loss)
            
            if self.avg_loss < best_loss - 0.000001:
                best_loss = self.avg_loss
                no_improve_epochs = 0 
            else:
                no_improve_epochs += 1
                
            self.perplexity = torch.exp(torch.tensor(self.avg_loss)).item()
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_description(f"Ep {epoch} | Loss:{self.avg_loss:.5f} | PPL:{self.perplexity:.4f} | LR:{current_lr:.1e} | Stall:{no_improve_epochs}/{self.patience_stop}")
            
            self.final_epoch = epoch 
            
            # --- 停止条件チェック ---
            if self.ppl_stop and self.perplexity < self.PPL_TARGET: 
                print(f"\n--- 目標達成 (暗記成功) ---")
                print(f"Epoch {epoch}: PPL {self.perplexity:.5f} < {self.PPL_TARGET}")
                break
            
            if no_improve_epochs >= self.patience_stop:
                print(f"\n--- 学習停滞 (暗記失敗と判断) ---")
                print(f"{self.patience_stop}エポック連続で改善なし。限界と判断し終了します。")
                print(f"最終PPL: {self.perplexity:.4f}, 最終LR: {current_lr:.1e}")
                break
            
            if epoch % self.snapshot_interval == 0:
                self._save_model(model, tokenizer, optimizer, is_final=False) 

        # --- 4. 最終保存 ---
        self._save_model(model, tokenizer, optimizer, is_final=True)
    
    def _save_model(self, model, tokenizer, optimizer, is_final=True):
        rounded_ppl = round(self.perplexity, 4)

        if is_final:
            model_folder_name = (
                f"{self.current_run_num:02d}_{self.today_date}" 
                f"_datasize_{self.data_size}"
                f"_epc_{self.final_epoch}"
                f"_ppl_{str(rounded_ppl).replace('.', '-')}" 
                f"_llm"
            )
        else:
            model_folder_name = (
                f"{self.current_run_num:02d}_{self.today_date}_snapshot_epc_{self.final_epoch}"
            )
            
        MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, model_folder_name)
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        
        checkpoint_data = {
            'epoch': self.final_epoch,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint_data, os.path.join(MODEL_DIR, 'optimizer_checkpoint.pt'))
        
        if is_final:
            self.model_folder_name = model_folder_name


# --- メイン実行ブロック ---
if __name__ == '__main__':
    
    filename = 'analyze/max_data_size/max_data_size.csv'
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    file_exists = os.path.isfile(filename)
    if not file_exists:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['embd', 'max_data_size', 'memorization_rate_at_limit'])

    # 実験パラメータ (Embedding Size)
    # ★修正1: n_embd=56から開始
    embd_list = [56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]
    
    # データサイズリスト
    data_size_list = (
        list(range(10, 100, 10)) + 
        list(range(100, 1000, 100)) + 
        list(range(1000, 3000, 250)) + 
        list(range(3000, 20001, 500))
    )
    data_size_list = sorted(list(set(data_size_list)))

    import gc
    
    # ★修正2: last_success_sizeを600に初期設定し、700から開始させる
    last_success_size = 600 

    for embd in embd_list:
        print(f"\n{'='*30}")
        print(f"開始: n_embd = {embd}")
        print(f"{'='*30}")
        
        data_row = []
        tmp_memorization_rate = 0.0
        
        start_index = 0
        if last_success_size > 0:
            try:
                # last_success_size=600 のインデックス + 1 = 700のインデックス
                start_index = data_size_list.index(last_success_size) + 1
            except ValueError:
                start_index = 0 
        
        if start_index >= len(data_size_list):
            print(f"n_embd={embd} はリスト内のすべてのサイズ ({data_size_list[-1]}) を暗記成功しています。")
            continue 

        for i in data_size_list[start_index:]:
            print(f"\n--- Testing Data Size: {i} (embd: {embd}) ---")
            
            torch.cuda.empty_cache()
            gc.collect()

            trainer = TRAIN_MODEL(
                model_base_dir='model/'+str(embd),
                data_dir='data/'+str(i),
                num_epochs=1000000, 
                learning_rate=1e-3, 
                patience=5000, 
                snapshot_interval=999999,
                ppl_target=1.001,
                ppl_stop=True,
                n_embd=embd,
                data_size=i
            )
            
            try:
                trainer.run_training()
            except FileNotFoundError as e:
                print(f"スキップ: {e}")
                
                final_max_size = last_success_size 
                final_rate = tmp_memorization_rate
                data_row = [embd, final_max_size, final_rate]
                
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
                break 
            
            print("検証中...")
            verify_memorization = VERIFY_MEMORYZATION(
                model_base_dir='model/'+str(embd),
                model_folder_input=trainer.model_folder_name,
                train_data_size=i,
                id_length=5, 
                colon_length=1, 
                data_length=32, 
                hash_length=12,
                prompt_length_body=6
            )
            
            memorization_rate = verify_memorization.verify_train_data()[1]
            print(f"暗記率: {memorization_rate}%")

            if memorization_rate < 100:
                print(f"× 暗記失敗 (Rate: {memorization_rate}%) -> 限界確定")
                
                final_max_size = last_success_size 
                final_rate = tmp_memorization_rate
                data_row = [embd, final_max_size, final_rate]
                
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
                break
            else:
                print(f"○ 暗記成功 (Rate: 100%) -> 次のサイズへ")
                last_success_size = i 
                tmp_memorization_rate = memorization_rate
                
                if i == data_size_list[-1]:
                    data_row = [embd, i, memorization_rate]
                    with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data_row)
                        
        if data_row:
             print(f"結果記録: embd={embd}, max_size={data_row[1]}")
        else:
             print(f"結果記録: embd={embd}, max_size={last_success_size}")

    print("\n全実験終了")