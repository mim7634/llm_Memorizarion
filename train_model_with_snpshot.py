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
from verify_memorization_splitBatch import VERIFY_MEMORYZATION

class TRAIN_MODEL:
    def __init__(self, data_dir='data', ppl_stop=True, ppl_target=1.001, num_epochs=1000000, learning_rate=1e-3, snapshot_interval=1000, model_base_dir='model', n_layer=4, n_head=8, n_embd=256, data_size="", patience=5000):
        # --- パスと設定 ---
        self.DATA_DIR = data_dir
        self.MODEL_BASE_DIR = model_base_dir
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        # ハイパーパラメータ
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 32
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
            numeric_prefixes = []
            for f in existing_folders:
                prefix = f[:2]
                if prefix.isdigit():
                    numeric_prefixes.append(int(prefix))
            if numeric_prefixes: 
                latest_num = max(numeric_prefixes)
        current_run_num = latest_num + 1
        return current_run_num, today_date 

    def _tokenize_function(self, examples, tokenizer):
        tokenized_input = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
        tokenized_input["labels"] = tokenized_input["input_ids"].copy()
        return tokenized_input

    def run_training(self):
        # --- 1. データ準備 ---
        tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if not os.path.exists(self.TRAIN_PATH):
            raise FileNotFoundError(f"データファイルが見つかりません: {self.TRAIN_PATH}")

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
            num_workers=0
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
        best_loss = float('inf')
        no_improve_epochs = 0
        
        progress_bar = tqdm(range(1, self.NUM_EPOCHS + 1), desc="Training")

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
            self.perplexity = torch.exp(torch.tensor(self.avg_loss)).item()
            self.final_epoch = epoch
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(self.avg_loss)

            if self.avg_loss < best_loss - 0.000001:
                best_loss = self.avg_loss
                no_improve_epochs = 0 
            else:
                no_improve_epochs += 1
            
            progress_bar.set_description(f"Ep {epoch} | Loss:{self.avg_loss:.5f} | PPL:{self.perplexity:.4f} | Stall:{no_improve_epochs}/{self.patience_stop}")
            
            # --- スナップショット & 暗記率チェック ---
            if epoch % self.snapshot_interval == 0:
                snapshot_folder = self._save_model(model, tokenizer, optimizer, is_final=False)
                
                # 中間検証の実行
                verifier = VERIFY_MEMORYZATION(
                    model_base_dir=self.MODEL_BASE_DIR,
                    model_folder_input=snapshot_folder,
                    train_data_size=self.data_size,
                    id_length=5, colon_length=1, data_length=32, hash_length=12, prompt_length_body=6
                )
                
                # ★修正ポイント：可変長引数を受け取れるようにアスタリスクを使用、または個別に受け取る
                verify_results = verifier.verify_train_data(debug=False)
                current_rate = verify_results[1] # インデックス1が暗記率であることを前提
                
                print(f"\n[Epoch {epoch}] Snapshot Verified: Memorization Rate = {current_rate}%")

                if current_rate >= 100.0:
                    print(f"--- 暗記率100%達成により早期終了 ---")
                    break

            # --- 停止条件チェック ---
            if self.ppl_stop and self.perplexity < self.PPL_TARGET: 
                print(f"\n--- PPL目標達成 ---")
                break
            
            if no_improve_epochs >= self.patience_stop:
                print(f"\n--- 学習停滞による終了 ---")
                break

        # --- 最終保存 ---
        self._save_model(model, tokenizer, optimizer, is_final=True)
    
    def _save_model(self, model, tokenizer, optimizer, is_final=True):
        rounded_ppl = round(self.perplexity, 4)
        if is_final:
            folder_name = (
                f"{self.current_run_num:02d}_{self.today_date}" 
                f"_datasize_{self.data_size}_epc_{self.final_epoch}_ppl_{str(rounded_ppl).replace('.', '-')}_llm"
            )
            self.model_folder_name = folder_name
        else:
            folder_name = f"{self.current_run_num:02d}_{self.today_date}_snapshot_epc_{self.final_epoch}"
            
        model_path = os.path.join(self.MODEL_BASE_DIR, folder_name)
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        checkpoint_data = {'epoch': self.final_epoch, 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint_data, os.path.join(model_path, 'optimizer_checkpoint.pt'))
        return folder_name

# --- メイン実行ブロック ---
if __name__ == '__main__':
    filename = 'analyze/max_data_size/max_data_size.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            csv.writer(f).writerow(['embd', 'max_data_size', 'memorization_rate_at_limit'])

    embd_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136]
    data_size_list = sorted(list(set(
        list(range(0, 100, 10)) + list(range(100, 1000, 100)) + 
        list(range(1000, 3000, 250)) + list(range(3000, 20001, 500))
    )))

    last_success_size = 30

    for embd in embd_list:
        print(f"\n{'='*30}\n開始: n_embd = {embd}\n{'='*30}")
        tmp_memorization_rate = 0.0
        
        try:
            start_index = data_size_list.index(last_success_size) + 1
        except ValueError:
            start_index = 0 

        for i in data_size_list[start_index:]:
            print(f"\n--- Testing Data Size: {i} (embd: {embd}) ---")
            torch.cuda.empty_cache()
            gc.collect()

            trainer = TRAIN_MODEL(
                model_base_dir='model/'+str(embd),
                data_dir='data/'+str(i),
                num_epochs=1000000, 
                learning_rate=1e-3, 
                patience=10000, 
                snapshot_interval=1000,
                ppl_target=1.001,
                n_embd=embd,
                data_size=i
            )
            
            try:
                trainer.run_training()
            except FileNotFoundError as e:
                print(f"スキップ: {e}")
                with open(filename, 'a', newline='') as f:
                    csv.writer(f).writerow([embd, last_success_size, tmp_memorization_rate])
                break 
            
            print("最終検証中...")
            verifier = VERIFY_MEMORYZATION(
                model_base_dir='model/'+str(embd),
                model_folder_input=trainer.model_folder_name,
                train_data_size=i,
                id_length=5, colon_length=1, data_length=32, hash_length=12, prompt_length_body=6
            )
            
            # ここでも戻り値のアンパックエラーを防ぐ
            final_verify_results = verifier.verify_train_data()
            memorization_rate = final_verify_results[1]
            
            print(f"最終暗記率: {memorization_rate}%")

            if memorization_rate < 100:
                print(f"× 暗記失敗 -> 限界確定")
                with open(filename, 'a', newline='') as f:
                    csv.writer(f).writerow([embd, last_success_size, tmp_memorization_rate])
                break
            else:
                print(f"○ 暗記成功 -> 次のサイズへ")
                last_success_size = i 
                tmp_memorization_rate = memorization_rate
                if i == data_size_list[-1]:
                    with open(filename, 'a', newline='') as f:
                        csv.writer(f).writerow([embd, i, memorization_rate])

    print("\n全実験終了")