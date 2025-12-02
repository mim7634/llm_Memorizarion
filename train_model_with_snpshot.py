import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, DefaultDataCollator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import datetime
import torch.nn.functional as F
import csv

# ユーザー環境の既存ファイルをインポート
from prepare_data import PREPARE_DATA
from verify_memorization import VERIFY_MEMORYZATION

class TRAIN_MODEL:
    def __init__(self, data_dir='data', ppl_stop=True, ppl_target=1.01, num_epochs=1000000, learning_rate=1e-5, snapshot_interval=999999, model_base_dir='model', n_layer=4, n_head=8, n_embd=256, data_size="", patience=50):
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
        self.NUM_PROCS = 1 
        
        # 訓練中に使用する変数
        self.ppl_stop = ppl_stop
        self.final_epoch = 0 
        self.avg_loss = 0
        self.perplexity = 0
        self.current_run_num, self.today_date = self._get_model_save_dir_prefix()
        self.model_folder_name = "" 
        self.data_size = data_size

        self.snapshot_interval = snapshot_interval
        
        # ★追加: Early Stopping用の設定
        self.patience = patience  # 何エポック改善がなければ諦めるか

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
        # print("1. データとトークナイザーを準備中...")
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
        # print("2. 小規模Transformerモデルを定義中...")
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
        
        # 学習再開ロジック（今回は基本未使用だが維持）
        start_epoch = 1
        if resume_from_dir:
            RESUME_DIR = os.path.join(self.MODEL_BASE_DIR, resume_from_dir)
            try:
                model = GPT2LMHeadModel.from_pretrained(RESUME_DIR).to(self.DEVICE)
                checkpoint = torch.load(os.path.join(RESUME_DIR, 'optimizer_checkpoint.pt'))
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
            except Exception as e:
                print(f"警告: リカバリ失敗。新規開始します。詳細: {e}")
                start_epoch = 1

        # --- 3. 訓練ループ ---
        # print(f"3. 訓練開始 (Max Epochs: {self.NUM_EPOCHS}, Target PPL: {self.PPL_TARGET})")
        model.train()
        
        # ★Early Stopping用変数
        best_loss = float('inf')
        no_improve_epochs = 0
        
        # tqdmの設定（動的に情報を更新するため）
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
            self.perplexity = torch.exp(torch.tensor(self.avg_loss)).item()
            
            # tqdmの表示を更新
            progress_bar.set_description(f"Epoch {epoch} | Loss: {self.avg_loss:.4f} | PPL: {self.perplexity:.4f} | Stall: {no_improve_epochs}/{self.patience}")
            
            self.final_epoch = epoch 
            
            # --- A. 目標達成チェック (PPL Stop) ---
            if self.ppl_stop:
                if self.perplexity < self.PPL_TARGET:
                    print(f"\n--- 目標達成 (暗記成功) ---")
                    print(f"Epoch {epoch}: PPL {self.perplexity:.4f} < {self.PPL_TARGET}")
                    break

            # --- B. 諦めチェック (Early Stopping) ---
            # Lossが前回ベストより有意(0.0001)に下がらなければカウント
            if self.avg_loss < best_loss - 0.0001:
                best_loss = self.avg_loss
                no_improve_epochs = 0 # リセット
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= self.patience:
                print(f"\n--- 学習停滞 (暗記失敗と判断) ---")
                print(f"{self.patience}エポック連続で改善なし。限界と判断し終了します。")
                print(f"最終PPL: {self.perplexity:.4f}")
                break
            
            # スナップショット保存（基本は実行されない設定）
            if epoch % self.snapshot_interval == 0:
                self._save_model(model, tokenizer, optimizer, is_final=False) 

        # --- 4. 訓練済みモデルの最終保存 ---
        self._save_model(model, tokenizer, optimizer, is_final=True)
    
    def _save_model(self, model, tokenizer, optimizer, is_final=True):
        """モデル保存"""
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
            # print(f"モデル保存完了: {MODEL_DIR}")


# --- メイン実行ブロック ---
if __name__ == '__main__':
    
    # 結果保存用CSVの設定
    filename = 'analyze/max_data_size/max_data_size.csv'
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # 既存のCSVがある場合、ヘッダー書き込みをスキップするかどうかの判定用
    file_exists = os.path.isfile(filename)
    
    if not file_exists:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['embd', 'max_data_size', 'memorization_rate_at_limit'])

    # 実験パラメータ
    # 時間短縮のため、少し刻みを粗くするか、範囲を絞ることを推奨しますが、元のリストを使用します。
    embd_list = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]
    
    # GPUメモリ確保のため、モデル定義前にガベージコレクション
    import gc

    for embd in embd_list:
        print(f"\n{'='*30}")
        print(f"開始: n_embd = {embd}")
        print(f"{'='*30}")
        
        data_row = []
        tmp_memorization_rate = 0.0
        last_success_size = 0
        
        # データサイズの刻み。限界探索のため 500 ずつ増加
        for i in range(500, 20001, 500): 
            print(f"\n--- Testing Data Size: {i} (embd: {embd}) ---")
            
            # メモリクリーンアップ
            torch.cuda.empty_cache()
            gc.collect()

            trainer = TRAIN_MODEL(
                model_base_dir='model/'+str(embd),
                data_dir='data/'+str(i),
                num_epochs=1000000,     # ★変更: 実質無限
                patience=50,            # ★追加: 50エポック停滞で諦める
                snapshot_interval=999999, # ★変更: 途中保存しない
                ppl_target=1.01,
                ppl_stop=True,
                learning_rate=2e-4,     # 高めの学習率
                n_embd=embd,
                data_size=i
            )
            
            # 学習実行
            trainer.run_training()
            
            # 検証実行
            print("検証中...")
            verify_memorization = VERIFY_MEMORYZATION(
                model_base_dir='model/'+str(embd),
                model_folder_input=trainer.model_folder_name,
                train_data_size=i
            )
            
            # verify_train_data()の戻り値が (bool, rate) であると仮定
            # [1] でレート(0-100)を取得
            memorization_rate = verify_memorization.verify_train_data()[1]
            print(f"暗記率: {memorization_rate}%")

            # ディスク容量節約のため、検証が終わったモデルは削除しても良いかもしれません
            # import shutil
            # shutil.rmtree(os.path.join('model', str(embd), trainer.model_folder_name))

            if memorization_rate < 100:
                print(f"× 暗記失敗 (Rate: {memorization_rate}%) -> 限界確定")
                # 今回失敗したので、記録するのは「前回成功したサイズ」
                # もし初回(500)で失敗したら 0 になる
                final_max_size = i - 500
                final_rate = tmp_memorization_rate
                
                # 初回で失敗した場合の特別処理
                if final_max_size < 0: 
                    final_max_size = 0
                    final_rate = 0
                
                data_row = [embd, final_max_size, final_rate]
                break
            else:
                print(f"○ 暗記成功 (Rate: 100%) -> サイズ増加")
                last_success_size = i
                tmp_memorization_rate = memorization_rate
                
                # 最大データサイズ(例:20000)まで行ってしまった場合
                if i == 20000:
                     data_row = [embd, i, memorization_rate]

        # 結果をCSVに書き込み
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
        
        print(f"結果記録: embd={embd}, max_size={data_row[1]}")

    print("\n全実験終了")