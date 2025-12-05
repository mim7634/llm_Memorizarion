import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
# 注: DDP環境下では、全てのプロセスがこれらのクラスにアクセスできる必要があります。
from prepare_data import PREPARE_DATA
from verify_memorization import VERIFY_MEMORYZATION

# Vast.aiの複数GPU環境で実行するために、DDPに対応したクラスに修正
class TRAIN_MODEL_DDP:
    def __init__(self, rank, world_size, data_dir='data', ppl_stop=True, ppl_target=1.001, num_epochs=1000000, learning_rate=1e-3, snapshot_interval=999999, model_base_dir='model', n_layer=4, n_head=8, n_embd=256, data_size="", patience=5000):
        # --- DDP 設定 ---
        self.rank = rank  # 現在のGPU ID (プロセスID)
        self.world_size = world_size # 全体のプロセス数 (GPU数)

        # --- パスと設定 ---
        self.DATA_DIR = data_dir
        self.MODEL_BASE_DIR = model_base_dir
        self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        # ハイパーパラメータ
        # rankをそのままGPU IDとして使用
        self.DEVICE = torch.device(f"cuda:{self.rank}") 
        self.BATCH_SIZE = 32 # 各GPUごとのバッチサイズ (全体では BATCH_SIZE * world_size)
        self.NUM_EPOCHS = num_epochs
        
        self.LEARNING_RATE = learning_rate 
        
        self.PPL_TARGET = ppl_target
        self.NUM_PROCS = 4 # データセットの準備用 (DDPとは別)
        
        # 訓練中に使用する変数
        self.ppl_stop = ppl_stop
        self.final_epoch = 0 
        self.avg_loss = 0
        self.perplexity = 0
        
        # メインプロセス (rank 0) のみでファイル操作/保存を行う
        if self.rank == 0:
            self.current_run_num, self.today_date = self._get_model_save_dir_prefix()
        else:
            self.current_run_num, self.today_date = None, None
            
        self.model_folder_name = "" 
        self.data_size = data_size
        self.snapshot_interval = snapshot_interval
        
        self.patience_stop = patience 

    def _get_model_save_dir_prefix(self):
        """モデル保存ディレクトリの連番生成 (rank 0 のみ実行)"""
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
        # ----------------------------------------------------
        # 1. データ準備
        # ----------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # rank 0 のみでトークナイザーの語彙サイズ変更を実行
        if self.rank == 0:
            self.TRAIN_PATH = os.path.join(self.DATA_DIR, "train_data.txt")
            if not os.path.exists(self.TRAIN_PATH):
                raise FileNotFoundError(f"データファイルが見つかりません: {self.TRAIN_PATH}\nデータサイズ {self.data_size} のファイルが存在するか確認してください。")
        
        # rank 0 のみがデータをロードし、他のプロセスは待機
        if self.rank == 0:
            raw_datasets = load_dataset("text", data_files={"train": self.TRAIN_PATH})
            tokenized_datasets = raw_datasets.map(
                lambda examples: self._tokenize_function(examples, tokenizer), 
                batched=True, 
                num_proc=self.NUM_PROCS, 
                remove_columns=["text"]
            )
        dist.barrier() # rank 0 の処理完了を待つ

        # 全プロセスでデータセットをロード（ファイルが見つからない場合はrank 0で既にエラーになっているはず）
        raw_datasets = load_dataset("text", data_files={"train": self.TRAIN_PATH})
        tokenized_datasets = raw_datasets.map(
            lambda examples: self._tokenize_function(examples, tokenizer), 
            batched=True, 
            num_proc=self.NUM_PROCS, 
            remove_columns=["text"]
        )

        data_collator = DefaultDataCollator()
        
        # DDPの核心: DistributedSampler を使用してデータを各GPUに均等に分割
        train_sampler = DistributedSampler(
            tokenized_datasets["train"],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        train_dataloader = DataLoader(
            tokenized_datasets["train"], 
            batch_size=self.BATCH_SIZE, # 各GPUごとのバッチサイズ
            sampler=train_sampler, # Samplerを指定
            collate_fn=data_collator,
            num_workers=4, 
            pin_memory=True # 高速化のため
        )
        
        # ----------------------------------------------------
        # 2. モデル定義
        # ----------------------------------------------------
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
        
        # DDPでモデルをラップ
        model = DDP(model, device_ids=[self.rank])
        
        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-7) 
        model.train()
        start_epoch = 1
        
        best_loss = float('inf')
        no_improve_epochs = 0
        
        # rank 0 のみプログレスバーを表示
        progress_bar = tqdm(range(start_epoch, self.NUM_EPOCHS + 1), desc="Training") if self.rank == 0 else range(start_epoch, self.NUM_EPOCHS + 1)

        # ----------------------------------------------------
        # 3. 学習ループ
        # ----------------------------------------------------
        for epoch in progress_bar:
            # DDPでは epoch ごとに sampler のシードを設定し、データのシャッフルを保証
            train_sampler.set_epoch(epoch)

            total_loss = 0
            for batch in train_dataloader:
                batch = {k: v.to(self.DEVICE) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            # 全プロセスからの損失を平均化 (reduce)
            avg_loss_tensor = torch.tensor([total_loss / len(train_dataloader)]).to(self.DEVICE)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM) # SUMで集約
            self.avg_loss = avg_loss_tensor.item() / self.world_size # world_sizeで割って平均を計算
            
            scheduler.step(self.avg_loss)
            
            # rank 0 のみでログ出力やチェックを行う
            if self.rank == 0:
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

        # DDPで学習停止が起きた場合、全プロセスに同期が必要
        if self.rank == 0:
            final_epoch = self.final_epoch
        else:
            final_epoch = 0 # 仮の値

        # final_epoch を全プロセスにブロードキャスト
        epoch_tensor = torch.tensor([final_epoch]).to(self.DEVICE)
        dist.broadcast(epoch_tensor, src=0)
        self.final_epoch = int(epoch_tensor.item())

        # 全プロセスで同期
        dist.barrier()

        # ----------------------------------------------------
        # 4. 最終保存 (rank 0 のみ)
        # ----------------------------------------------------
        if self.rank == 0:
            # DDPモデルから内部のモジュールを取り出して保存
            self._save_model(model.module, tokenizer, optimizer, is_final=True)
            return True # rank 0 のみ実験継続
        else:
            return False # rank > 0 は実験ループを継続しない

    # モデル保存関数 (rank 0 のみ実行)
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
            # DDPモデルは 'module' 属性を持っているので、optimizerの状態保存には注意が必要ですが、
            # ここではDDPをラップする前の optimizer を使用しているため、state_dictで問題なし
            'optimizer_state_dict': optimizer.state_dict(), 
        }
        torch.save(checkpoint_data, os.path.join(MODEL_DIR, 'optimizer_checkpoint.pt'))
        
        if is_final:
            self.model_folder_name = model_folder_name

# ----------------------------------------------------
# DDP起動用ラッパー関数
# ----------------------------------------------------
def _run_experiment(rank, world_size, embd, data_size_list, last_success_size, filename):
    
    # DDPの初期化
    os.environ['MASTER_ADDR'] = 'localhost' # Vast.aiの単一インスタンス内の複数GPUを想定
    os.environ['MASTER_PORT'] = '12355' # 任意の空きポート
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # rank 0 のみでログファイルの設定
    if rank == 0:
        print(f"\n{'='*30}")
        print(f"開始: n_embd = {embd} (全GPU数: {world_size})")
        print(f"{'='*30}")
    
    torch.cuda.set_device(rank) # プロセスごとに使用するGPUを設定
    
    tmp_memorization_rate = 0.0
    
    start_index = 0
    if last_success_size > 0:
        try:
            start_index = data_size_list.index(last_success_size) + 1
        except ValueError:
            start_index = 0 
    
    if rank == 0 and start_index >= len(data_size_list):
        print(f"n_embd={embd} はリスト内のすべてのサイズ ({data_size_list[-1]}) を暗記成功しています。")
        dist.destroy_process_group()
        return

    
    # 各プロセスが同時に次のデータサイズに進むのを避けるため、rank 0 でのみループを回し、他のプロセスは待機
    # ただし、DDPでは学習自体が並列なので、ここではrank 0 が主導する形で実験を制御
    
    current_last_success_size = last_success_size # このembdでの最後の成功サイズを追跡
    
    for i in data_size_list[start_index:]:
        
        if rank == 0:
            print(f"\n--- Testing Data Size: {i} (embd: {embd}, GPU: {world_size}基) ---")
        
        # 全プロセスで同期: データサイズ変更を全プロセスで揃える
        dist.barrier() 

        # GPUメモリのクリア
        torch.cuda.empty_cache()
        gc.collect()

        trainer = TRAIN_MODEL_DDP(
            rank=rank,
            world_size=world_size,
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
            # run_training は rank 0 のみ True を返す
            is_master_to_continue = trainer.run_training() 
        except FileNotFoundError as e:
            if rank == 0:
                print(f"スキップ: {e}")
                
                final_max_size = current_last_success_size 
                final_rate = tmp_memorization_rate
                data_row = [embd, final_max_size, final_rate]
                
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
                
                # FileNotFoundErrorは全プロセスで同時に発生するとは限らないが、ここではrank 0で制御
                dist.destroy_process_group()
                return # 実験終了

        # rank 0 のみが検証を実行
        if rank == 0:
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
                
                final_max_size = current_last_success_size 
                final_rate = tmp_memorization_rate
                data_row = [embd, final_max_size, final_rate]
                
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
                
                print(f"結果記録: embd={embd}, max_size={final_max_size}")
                # 暗記失敗: ループを抜けるフラグを全プロセスに送信するためにブロードキャスト
                break_loop_flag = torch.tensor([1]).to(rank) 
                
            else:
                print(f"○ 暗記成功 (Rate: 100%) -> 次のサイズへ")
                current_last_success_size = i 
                tmp_memorization_rate = memorization_rate
                
                if i == data_size_list[-1]:
                    data_row = [embd, i, memorization_rate]
                    with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data_row)
                    
                    print(f"結果記録: embd={embd}, max_size={i}")
                    break_loop_flag = torch.tensor([1]).to(rank)
                else:
                    break_loop_flag = torch.tensor([0]).to(rank)
        else:
            # rank > 0 のプロセスは rank 0 からのフラグを待つ
            break_loop_flag = torch.tensor([0]).to(rank)

        # ループを抜けるフラグを全プロセスにブロードキャスト
        dist.broadcast(break_loop_flag, src=0)
        if break_loop_flag.item() == 1:
            break

    # DDP環境のクリーンアップ
    dist.destroy_process_group()


# ----------------------------------------------------
# メイン実行ブロック
# ----------------------------------------------------
if __name__ == '__main__':
    
    # Vast.aiのインスタンスで利用可能なGPUの数を取得
    # 環境変数 'CUDA_VISIBLE_DEVICES' などから自動取得可能だが、ここでは明示的に12機を想定
    NUM_GPUS = 12 
    
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
    embd_list = [56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]
    
    # データサイズリスト
    data_size_list = (
        list(range(10, 100, 10)) + 
        list(range(100, 1000, 100)) + 
        list(range(1000, 3000, 250)) + 
        list(range(3000, 20001, 500))
    )
    data_size_list = sorted(list(set(data_size_list)))

    # last_success_sizeを600に初期設定
    last_success_size = 600 

    for embd in embd_list:
        # torch.multiprocessing.spawn を使用して、各embdごとにDDPプロセスを起動
        # func: _run_experiment (各プロセスで実行する関数)
        # args: (world_size, embd, data_size_list, last_success_size, filename)
        # nprocs: NUM_GPUS (起動するプロセス/GPU数)
        try:
            mp.spawn(
                _run_experiment,
                args=(NUM_GPUS, embd, data_size_list, last_success_size, filename),
                nprocs=NUM_GPUS,
                join=True # 全プロセスが終了するまで待機
            )
        except Exception as e:
            print(f"DDPプロセスでエラーが発生しました (embd={embd}): {e}")
            # エラー発生時も次のembdのテストに進む
        
    print("\n全実験終了")