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
        self.DATA_DIR = data_dir
        self.MODEL_BASE_DIR = model_base_dir
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = num_epochs
        self.LEARNING_RATE = learning_rate 
        self.PPL_TARGET = ppl_target
        self.NUM_PROCS = 4 
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
        today_date = datetime.date.today().strftime("%Y%m%d")
        os.makedirs(self.MODEL_BASE_DIR, exist_ok=True)
        existing_folders = [d for d in os.listdir(self.MODEL_BASE_DIR) if os.path.isdir(os.path.join(self.MODEL_BASE_DIR, d))]
        latest_num = 0
        if existing_folders:
            numeric_prefixes = [int(f[:2]) for f in existing_folders if f[:2].isdigit()]
            if numeric_prefixes: latest_num = max(numeric_prefixes)
        return latest_num + 1, today_date 

    def _tokenize_function(self, examples, tokenizer):
        tokenized_input = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
        tokenized_input["labels"] = tokenized_input["input_ids"].copy()
        return tokenized_input

    def run_training(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if not os.path.exists(self.TRAIN_PATH):
            raise FileNotFoundError(f"データファイルが見つかりません: {self.TRAIN_PATH}")

        raw_datasets = load_dataset("text", data_files={"train": self.TRAIN_PATH})
        tokenized_datasets = raw_datasets.map(lambda examples: self._tokenize_function(examples, tokenizer), batched=True, num_proc=self.NUM_PROCS, remove_columns=["text"])
        
        data_collator = DefaultDataCollator()
        train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=self.BATCH_SIZE, collate_fn=data_collator, shuffle=True)
        
        MODEL_CONFIG = GPT2Config(vocab_size=len(tokenizer), n_layer=self.n_layer, n_head=self.n_head, n_embd=self.n_embd, pad_token_id=tokenizer.pad_token_id, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
        model = GPT2LMHeadModel(MODEL_CONFIG)
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
            scheduler.step(self.avg_loss)

            if self.avg_loss < best_loss - 0.000001:
                best_loss = self.avg_loss
                no_improve_epochs = 0 
            else:
                no_improve_epochs += 1
            
            progress_bar.set_description(f"Ep {epoch} | Loss:{self.avg_loss:.5f} | PPL:{self.perplexity:.4f} | Stall:{no_improve_epochs}/{self.patience_stop}")
            
            if epoch % self.snapshot_interval == 0:
                snap_folder = self._save_model(model, tokenizer, optimizer, is_final=False)
                verifier = VERIFY_MEMORYZATION(model_base_dir=self.MODEL_BASE_DIR, 
                                               model_folder_input=snap_folder, 
                                               train_data_size=self.data_size, 
                                               id_length=5, colon_length=1, 
                                               data_length=32, 
                                               hash_length=12, 
                                               prompt_length_body=6
                            )
                current_rate = verifier.verify_train_data(debug=False)[1]
                print(f"\n[Epoch {epoch}] Snapshot Rate: {current_rate}%")
                if current_rate >= 100.0:
                    print("--- 暗記率100%達成による早期終了 ---")
                    break

            if (self.ppl_stop and self.perplexity < self.PPL_TARGET) or (no_improve_epochs >= self.patience_stop):
                print(f"\n--- 停止条件到達 (PPL:{self.perplexity:.4f}, Stall:{no_improve_epochs}) ---")
                break

        self._save_model(model, tokenizer, optimizer, is_final=True)

    def _save_model(self, model, tokenizer, optimizer, is_final=True):
        rounded_ppl = round(self.perplexity, 4)
        if is_final:
            folder_name = f"{self.current_run_num:02d}_{self.today_date}_datasize_{self.data_size}_epc_{self.final_epoch}_ppl_{str(rounded_ppl).replace('.', '-')}_llm"
            self.model_folder_name = folder_name
        else:
            folder_name = f"{self.current_run_num:02d}_{self.today_date}_snapshot_epc_{self.final_epoch}"
        
        model_path = os.path.join(self.MODEL_BASE_DIR, folder_name)
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        torch.save({'epoch': self.final_epoch, 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_path, 'optimizer_checkpoint.pt'))
        return folder_name

def execute_trial(embd, size):
    """単一試行の実行と結果判定"""
    print(f"\n{'='*20} Testing Size: {size} (embd: {embd}) {'='*20}")
    torch.cuda.empty_cache()
    gc.collect()
    
    trainer = TRAIN_MODEL(model_base_dir=f'model/{embd}', 
                          data_dir=f'data/{size}', 
                          num_epochs=100000, 
                          learning_rate=1e-3, 
                          patience=10000, 
                          snapshot_interval=1000, 
                          ppl_target=1.001, 
                          n_embd=embd,
                          data_size=size
            )
    
    try:
        trainer.run_training()
        print("最終検証中...")
        verifier = VERIFY_MEMORYZATION(
            model_base_dir=f'model/{embd}',
            model_folder_input=trainer.model_folder_name, 
            train_data_size=size, id_length=5, 
            colon_length=1, data_length=32, 
            hash_length=12, 
            prompt_length_body=6
        )
        rate = verifier.verify_train_data()[1]
        print(f"最終暗記率: {rate}%")
        return rate >= 100.0, rate
    except FileNotFoundError as e:
        print(f"スキップ: {e}")
        return None, 0.0

if __name__ == '__main__':
    filename = 'analyze/max_data_size/max_data_size.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            csv.writer(f).writerow(['embd', 'max_data_size', 'rate'])

    embd_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136]
    data_size_list = sorted(list(set(list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 3000, 250)) + list(range(3000, 20001, 500)))))

    for embd in embd_list:
        print(f"\n\n{'#'*50}\n開始: n_embd = {embd}\n{'#'*50}")
        low_idx = 0
        high_idx = len(data_size_list) - 1
        last_success_size = 0
        tmp_rate = 0.0

        # 1. 指数探索フェーズ
        step = 1
        curr_idx = 0
        while curr_idx < len(data_size_list):
            success, rate = execute_trial(embd, data_size_list[curr_idx])
            if success is None: break # ファイルなし
            
            if success:
                print(f"○ 暗記成功 -> 次はステップ {step*2} でジャンプ")
                last_success_size = data_size_list[curr_idx]
                tmp_rate = rate
                low_idx = curr_idx
                curr_idx += step
                step *= 2
            else:
                print(f"× 暗記失敗 -> 二分探索へ移行")
                high_idx = curr_idx
                break
            if curr_idx >= len(data_size_list): high_idx = len(data_size_list) - 1

        # 2. 二分探索フェーズ
        while high_idx - low_idx > 1:
            mid_idx = (low_idx + high_idx) // 2
            success, rate = execute_trial(embd, data_size_list[mid_idx])
            if success:
                print(f"○ (二分) 成功")
                low_idx = mid_idx
                last_success_size = data_size_list[mid_idx]
                tmp_rate = rate
            else:
                print(f"× (二分) 失敗")
                high_idx = mid_idx

        print(f"\n[RESULT] embd:{embd} Max Size:{last_success_size} (Rate:{tmp_rate}%)")
        with open(filename, 'a', newline='') as f:
            csv.writer(f).writerow([embd, last_success_size, tmp_rate])

    print("\n全実験終了")