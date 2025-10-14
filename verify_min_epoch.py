import csv
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, DefaultDataCollator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import datetime
import torch.nn.functional as F

import hashlib
import numpy as np

from verify_memorization import VERIFY_MEMORYZATION
from train_model import TRAIN_MODEL

epoch = 10

file_name = 'relation_epoch_and_memorization_rate.csv'
header = ['model_name', 'epoch', 'memorization_rate', 'avg_ppl', 'avg_eval_ppl']
data = []

if __name__ == '__main__':
    while True:
        trainer = TRAIN_MODEL(num_epochs=epoch)
        trainer.run_training()
        model_folder_name = trainer.model_folder_name

        obj = VERIFY_MEMORYZATION(model_folder_input=model_folder_name)
        model_folder_input, memorization_rate, avg_ppl = obj.verify()

        print("\n---------------------------------------------------")
        print("訓練データの暗記検証結果:")
        print(f"検証対象モデル: {model_folder_input}")
        print(f"暗記率: {memorization_rate:.2f}%")
        print(f"平均PPL: {avg_ppl:.4f} (1.0に近いほど完璧な暗記)")
        print("---------------------------------------------------")

        # --- 4. 汎化能力の検証（検証データ PPL） ---
        print("\n--- 汎化能力の検証（未見データ）---")
        with open(obj.EVAL_DATA_PATH, "r", encoding="utf-8") as f:
            eval_data_lines = [line.strip() for line in f.readlines()]
            
        eval_ppls = []
        with torch.no_grad():
            for text in tqdm(eval_data_lines, desc="検証データPPL計算中"):
                full_input = obj.loaded_tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True).to(obj.DEVICE)
                labels = full_input.clone()
                loss = obj.loaded_model(full_input, labels=labels).loss.item()
                eval_ppls.append(np.exp(loss))

        avg_eval_ppl = np.mean(eval_ppls)
        print(f"未見の検証データ平均PPL: {avg_eval_ppl:.4f}")
        print("（暗記データPPLと比較して高い値であれば、汎化能力がないことが証明されます）")
        print("---------------------------------------------------")

        data.append([
            model_folder_name,
            epoch,
            memorization_rate,
            avg_ppl,
            avg_eval_ppl
        ])

        if memorization_rate == 75.56:
            print("MIN_EPOCH --->", epoch - 10)
            with open(file_name, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                writer.writerow(header)
                writer.writerows(data)
            print(f"ファイル '{file_name}' が正常に作成されました。")

            break

        epoch += 10

